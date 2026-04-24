// FSDP Memory Estimator
// All sizes are returned in bytes; UI converts to GB (1 GiB = 2^30).

const GB = 1024 ** 3;
const MB = 1024 ** 2;

// ---------------------------------------------------------------------------
// Parameter counting
// ---------------------------------------------------------------------------

function countParams(m) {
  // m is a model config object (see presets.js for fields)
  const head_dim = m.head_dim || (m.hidden_size / m.num_heads);
  const h = m.hidden_size;
  const i = m.intermediate_size;
  const L = m.num_layers;
  const v = m.vocab_size;
  const num_q = m.num_heads;
  const num_kv = m.num_kv_heads || m.num_heads;

  // Attention: q, k, v, o
  const q_proj = h * num_q * head_dim;
  const k_proj = h * num_kv * head_dim;
  const v_proj = h * num_kv * head_dim;
  const o_proj = num_q * head_dim * h;
  const attn = q_proj + k_proj + v_proj + o_proj;

  // MLP (SwiGLU): gate, up, down
  const mlp = 2 * h * i + i * h;

  // RMSNorm: 2 per layer (input + post-attn) — negligible but we count
  const norms = 2 * h;

  const per_layer = attn + mlp + norms;

  const embed = v * h;
  const lm_head = m.tie_embeddings ? 0 : v * h;
  const final_norm = h;

  const total = L * per_layer + embed + lm_head + final_norm;
  return { total, per_layer, embed, lm_head };
}

// ---------------------------------------------------------------------------
// FSDP memory model
// ---------------------------------------------------------------------------

// optimizer state bytes per fp32 parameter (master weights are tracked separately)
const OPTIM_STATE_BYTES = {
  adamw:        8,   // m + v (each fp32)
  adam:         8,
  sgd_momentum: 4,   // momentum buffer
  sgd:          0,
  adam_8bit:    2,   // bnb quantised m+v
};

const PRECISION_BYTES = { fp32: 4, fp16: 2, bf16: 2 };

function estimate(cfg) {
  const m = cfg.model;
  const counts = countParams(m);
  const P = counts.total;

  const N = cfg.world_size;
  const compute_bytes  = PRECISION_BYTES[cfg.compute_dtype];
  const grad_bytes_lp  = compute_bytes;             // gradients in low-prec (cast on reduce)
  const master_bytes   = 4;                         // fp32 master copy held by optimizer
  const optim_bytes    = OPTIM_STATE_BYTES[cfg.optimizer];

  // Sharding factor by strategy
  // FULL_SHARD       — params, grads, optim all sharded across N
  // SHARD_GRAD_OP    — params replicated, grads + optim sharded across N
  // HYBRID_SHARD     — params/grads/optim sharded within shard_size, replicated across replicas
  // NO_SHARD / DDP   — everything replicated
  let shard_param = 1, shard_grad = 1, shard_optim = 1;
  if (cfg.strategy === "FULL_SHARD") {
    shard_param = N; shard_grad = N; shard_optim = N;
  } else if (cfg.strategy === "SHARD_GRAD_OP") {
    shard_param = 1; shard_grad = N; shard_optim = N;
  } else if (cfg.strategy === "HYBRID_SHARD") {
    const s = Math.min(cfg.shard_size || 8, N);
    shard_param = s; shard_grad = s; shard_optim = s;
  } else { /* DDP */
    shard_param = 1; shard_grad = 1; shard_optim = 1;
  }

  // Persistent state, per GPU (bytes)
  const params_mem = compute_bytes * P / shard_param;
  const grads_mem  = grad_bytes_lp * P / shard_grad;
  // Master weights only exist if mixed precision (compute != fp32) or bf16 master not in use
  const master_mem = (cfg.compute_dtype === "fp32") ? 0 : (master_bytes * P / shard_optim);
  const optim_mem  = optim_bytes * P / shard_optim;
  const optimizer_total = master_mem + optim_mem;

  // Communication / all-gather buffer
  // Default FSDP unit = one transformer block. Peak transient = 2 blocks worth in low-prec
  // (current block being computed + next being prefetched).
  const block_params = counts.per_layer;
  // For DDP there is no all-gather of parameters
  const comm_mem = (cfg.strategy === "NO_SHARD") ? 0
                 : compute_bytes * block_params * 2;

  // ---------------- Activations ----------------
  const s = cfg.seq_len;
  const b = cfg.micro_batch;
  const h = m.hidden_size;
  const a = m.num_heads;
  const L = m.num_layers;
  const i = m.intermediate_size;
  const v = m.vocab_size;
  const act_dtype_bytes = compute_bytes; // activations stored in compute dtype

  // Per-layer activation memory (Korthikanti et al. 2022, no TP, no SP).
  //   sbh × 2 bytes per saved tensor, ~17 saved tensors per block ⇒ sbh·34 (for bf16)
  // Plus attention scores (s²·b·a·2) — skipped when FlashAttention is on.
  let act_per_layer = s * b * h * 17 * act_dtype_bytes / 2; // 17 sbh in fp16/bf16 ≈ sbh·34
  // intermediate-state for the MLP up/gate (i is often ~4h, but be precise)
  act_per_layer += s * b * i * 2 * act_dtype_bytes; // up + gate activations

  if (!cfg.flash_attn) {
    act_per_layer += s * s * b * a * act_dtype_bytes; // attention scores
  }

  let act_mem = 0;
  if (cfg.act_ckpt === "none") {
    act_mem = L * act_per_layer;
  } else if (cfg.act_ckpt === "selective") {
    // Keep MLP/norm but drop attention scores and softmax intermediates.
    // Roughly 60% of full activation memory.
    act_mem = L * act_per_layer * 0.6;
  } else if (cfg.act_ckpt === "full") {
    // Only the layer-input residual stream is retained for each block.
    act_mem = L * s * b * h * act_dtype_bytes  // residual inputs
            + act_per_layer;                    // one layer recomputed at a time
  }

  // Logits + loss activations (often dominate at large vocab × long seq)
  // logits: s·b·V in compute dtype
  // softmax + loss intermediates: often kept in fp32 for stability
  const logits_mem = s * b * v * act_dtype_bytes;
  const loss_mem   = s * b * v * 4; // fp32 cross-entropy
  // If activation checkpointing wraps the lm_head it can be reduced; assume not by default.
  let head_act_mem = logits_mem + loss_mem;
  if (cfg.ckpt_lm_head) head_act_mem = logits_mem; // saved logits only

  act_mem += head_act_mem;

  // Embedding activations (s·b·h, small)
  act_mem += s * b * h * act_dtype_bytes;

  // ---------------- Framework / misc ----------------
  // PyTorch + CUDA context, NCCL buffers, cuDNN workspace, allocator fragmentation.
  // Empirically ~1.0–2.0 GB on H100/A100.
  const overhead = 1.5 * GB + 0.5 * GB; // baseline + workspace

  const total = params_mem + grads_mem + optimizer_total + act_mem + comm_mem + overhead;

  return {
    P,
    block_params,
    components: {
      "Parameters":       params_mem,
      "Gradients":        grads_mem,
      "Optimizer states": optimizer_total,
      "Activations":      act_mem,
      "Comm buffers":     comm_mem,
      "Framework":        overhead,
    },
    total,
  };
}

// ---------------------------------------------------------------------------
// UI wiring
// ---------------------------------------------------------------------------

const $  = (id) => document.getElementById(id);
const fmtGB = (bytes) => (bytes / GB).toFixed(2);
const fmtParams = (n) => {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(2) + "K";
  return n.toString();
};

let breakdownChart, pieChart, sweepChart;

// Tonal palette: ordered by component dominance.
// Activations & Optimizer typically dominate, so they get the accent shades;
// the rest fade toward neutral grays.
const COLORS = {
  "Activations":      "#c14a16",   // accent terracotta
  "Optimizer states": "#d97f4c",   // accent · light
  "Parameters":       "#a89478",   // warm tan
  "Gradients":        "#806f5a",   // muted brown
  "Framework":        "#a39a8a",   // ink-faint
  "Comm buffers":     "#c9c0ae",   // soft
};
const INK         = "#1a1814";
const INK_MUTE    = "#6d6457";
const INK_FAINT   = "#a39a8a";
const RULE        = "#d9d2c2";
const RULE_SOFT   = "rgba(26,24,20,0.10)";
const ACCENT      = "#c14a16";
const ACCENT_DEEP = "#8d3209";
const PAPER       = "#f4f0e8";

// Apply Chart.js global defaults once.
if (window.Chart) {
  Chart.defaults.font.family = '"Newsreader", "Iowan Old Style", Georgia, serif';
  Chart.defaults.font.size   = 12;
  Chart.defaults.color       = INK_MUTE;
  Chart.defaults.borderColor = RULE;
  Chart.defaults.animation   = { duration: 360, easing: "easeOutQuart" };
}

// Pretty model name + a short prose summary that gives the headline number context.
function describeRun(cfg, result) {
  const presetName = $("model-preset").value;
  const modelLabel = (presetName === "custom") ? "your custom model" : presetName;
  const strat = {
    FULL_SHARD:   "FULL_SHARD",
    SHARD_GRAD_OP:"SHARD_GRAD_OP",
    HYBRID_SHARD: "HYBRID_SHARD",
    NO_SHARD:     "DDP",
  }[cfg.strategy] || cfg.strategy;
  const ckpt = cfg.act_ckpt === "none" ? "no checkpointing" : `${cfg.act_ckpt} checkpointing`;
  const fa   = cfg.flash_attn ? "FlashAttention on" : "FlashAttention off";
  const opt  = ({ adamw: "AdamW", adam: "Adam", sgd_momentum: "SGD+momentum", sgd: "SGD", adam_8bit: "Adam 8-bit" })[cfg.optimizer];
  const gpu  = window.GPU_PRESETS[cfg.gpu]?.name || cfg.gpu;
  return (
    `Training <strong>${modelLabel}</strong> on <strong>${cfg.world_size}× ${gpu}</strong> ` +
    `with <strong>${strat}</strong>, <strong>${cfg.compute_dtype}</strong> precision and <strong>${opt}</strong>, ` +
    `at micro-batch <strong>${cfg.micro_batch}</strong> × seq <strong>${cfg.seq_len}</strong>, ${ckpt}, ${fa}.`
  );
}

function readConfig() {
  const presetName = $("model-preset").value;
  let model;
  if (presetName === "custom") {
    model = {
      hidden_size:       +$("hidden_size").value,
      intermediate_size: +$("intermediate_size").value,
      num_layers:        +$("num_layers").value,
      num_heads:         +$("num_heads").value,
      num_kv_heads:      +$("num_kv_heads").value,
      vocab_size:        +$("vocab_size").value,
      tie_embeddings:    $("tie_embeddings").checked,
    };
  } else {
    model = window.MODEL_PRESETS[presetName];
  }
  return {
    model,
    world_size:    +$("world_size").value,
    micro_batch:   +$("micro_batch").value,
    seq_len:       +$("seq_len").value,
    compute_dtype: $("compute_dtype").value,
    optimizer:    $("optimizer").value,
    strategy:     $("strategy").value,
    shard_size:   +$("shard_size").value,
    act_ckpt:     $("act_ckpt").value,
    flash_attn:   $("flash_attn").checked,
    ckpt_lm_head: $("ckpt_lm_head").checked,
    gpu:          $("gpu").value,
  };
}

function syncPresetFields() {
  const presetName = $("model-preset").value;
  const customFields = $("custom-fields");
  if (presetName === "custom") {
    customFields.style.display = "";
    return;
  }
  customFields.style.display = "none";
  const m = window.MODEL_PRESETS[presetName];
  $("hidden_size").value       = m.hidden_size;
  $("intermediate_size").value = m.intermediate_size;
  $("num_layers").value        = m.num_layers;
  $("num_heads").value         = m.num_heads;
  $("num_kv_heads").value      = m.num_kv_heads || m.num_heads;
  $("vocab_size").value        = m.vocab_size;
  $("tie_embeddings").checked  = !!m.tie_embeddings;
}

function syncStrategyFields() {
  const strat = $("strategy").value;
  $("shard-size-row").style.display = (strat === "HYBRID_SHARD") ? "" : "none";
}

function renderResult(cfg, result) {
  // ── Headline figure ──
  $("kpi-total").textContent = fmtGB(result.total);
  $("caption").innerHTML     = describeRun(cfg, result);

  // GPU fit
  const gpu = window.GPU_PRESETS[cfg.gpu];
  const totalGB = result.total / GB;
  const fitEl = $("kpi-fit");
  const barEl = $("fit-bar");
  if (gpu) {
    const pct = (totalGB / gpu.mem_gb) * 100;
    const cls = pct < 70 ? "" : pct < 95 ? "warn" : "bad";
    fitEl.innerHTML =
      `<span><em>${fmtGB(result.total)} GB</em> of <em>${gpu.mem_gb} GB</em> available on each ${gpu.name}.</span>` +
      `<span class="end">${pct.toFixed(0)}%</span>`;
    barEl.style.width = Math.min(100, pct).toFixed(1) + "%";
    barEl.className = "fit-line-fill " + cls;
  } else {
    fitEl.textContent = "—";
    barEl.style.width = "0%";
    barEl.className = "fit-line-fill";
  }

  // ── Ledger table ──
  const ordered = Object.entries(result.components)
    .sort((a, b) => b[1] - a[1]); // largest first
  const tbody = $("component-table");
  tbody.innerHTML = "";
  for (const [k, v] of ordered) {
    const pct = (v / result.total * 100).toFixed(1);
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><span class="swatch" style="background:${COLORS[k]}"></span>${k}</td>
      <td class="num">${fmtGB(v)}</td>
      <td class="num">${pct}%</td>`;
    tbody.appendChild(tr);
  }
  $("ledger-total").textContent = fmtGB(result.total);

  // ── Chart palette (matches ledger order) ──
  const labels = ordered.map(([k]) => k);
  const data   = ordered.map(([, v]) => v / GB);
  const colors = labels.map(l => COLORS[l]);

  const axisX = {
    grid:   { color: RULE_SOFT, drawTicks: false },
    ticks:  { color: INK_MUTE, font: { size: 11, family: '"Newsreader", serif', style: "italic" }, padding: 6 },
    border: { color: RULE },
  };
  const axisY = {
    grid:   { color: RULE_SOFT, drawTicks: false },
    ticks:  { color: INK_MUTE, font: { size: 11, family: '"Newsreader", serif', style: "italic" }, padding: 8 },
    border: { color: RULE },
  };

  // ── Slim horizontal stacked bar (single row, no legend, no axes) ──
  breakdownChart = safeChart(breakdownChart, $("chart-breakdown"), {
    type: "bar",
    data: {
      labels: [""],
      datasets: labels.map((l, idx) => ({
        label: l,
        data: [data[idx]],
        backgroundColor: colors[idx],
        borderColor: PAPER,
        borderWidth: 1,
        borderSkipped: false,
        barPercentage: 1,
        categoryPercentage: 1,
      })),
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      layout: { padding: 0 },
      scales: {
        x: { stacked: true, display: false, grid: { display: false } },
        y: { stacked: true, display: false, grid: { display: false } },
      },
      plugins: {
        tooltip: tooltipStyle((ctx) => `${ctx.dataset.label} — ${ctx.parsed.x.toFixed(2)} GB`),
        legend:  { display: false },
      },
    },
  });

  // ── Doughnut, monochrome-ish ──
  pieChart = safeChart(pieChart, $("chart-pie"), {
    type: "doughnut",
    data: {
      labels,
      datasets: [{
        data,
        backgroundColor: colors,
        borderColor: PAPER,
        borderWidth: 2,
        hoverBorderColor: PAPER,
        hoverOffset: 4,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "65%",
      plugins: {
        tooltip: tooltipStyle((ctx) => `${ctx.label} — ${ctx.parsed.toFixed(2)} GB`),
        legend:  { display: false },
      },
    },
  });

  // ── Sweep curve — single thin terracotta line, dashed reference ──
  const sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
  const sweepData = sizes.map(N => {
    try { return estimate({ ...cfg, world_size: N }).total / GB; }
    catch (_) { return null; } // tolerate transient errors so sweep still draws
  });

  sweepChart = safeChart(sweepChart, $("chart-sweep"), {
    type: "line",
    data: {
      labels: sizes,
      datasets: [
        {
          label: "per-GPU peak",
          data: sweepData,
          borderColor: ACCENT,
          backgroundColor: "rgba(193,74,22,0.06)",
          borderWidth: 1.4,
          tension: 0.3,
          fill: true,
          pointRadius: 3,
          pointBackgroundColor: PAPER,
          pointBorderColor: ACCENT,
          pointBorderWidth: 1.2,
          pointHoverRadius: 5,
        },
        ...(gpu ? [{
          label: `${gpu.name} ceiling`,
          data: sizes.map(() => gpu.mem_gb),
          borderColor: INK_MUTE,
          borderDash: [3, 3],
          borderWidth: 1,
          pointRadius: 0,
          fill: false,
        }] : []),
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      layout: { padding: { top: 6, right: 4, bottom: 0 } },
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { ...axisX, title: { display: true, text: "world size", color: INK_MUTE, font: { size: 12, family: '"Newsreader", serif', style: "italic" }, padding: { top: 8 } } },
        y: { ...axisY, beginAtZero: true, title: { display: true, text: "GB per GPU", color: INK_MUTE, font: { size: 12, family: '"Newsreader", serif', style: "italic" } } },
      },
      plugins: {
        tooltip: tooltipStyle((ctx) => `${ctx.parsed.y.toFixed(2)} GB at ${ctx.parsed.x} GPUs`),
        legend:  {
          display: true,
          position: "bottom",
          align: "start",
          labels: {
            color: INK_MUTE,
            font: { family: '"Newsreader", serif', size: 12, style: "italic" },
            boxWidth: 16,
            boxHeight: 8,
            padding: 14,
            usePointStyle: false,
          },
        },
      },
    },
  });
}

// Construct a Chart.js chart on `canvas`, replacing `prev` if present.
// Failures in one chart never kill subsequent charts. Errors surface in the
// page-level error region so users notice without opening DevTools.
function safeChart(prev, canvas, config) {
  try {
    if (prev) prev.destroy();
  } catch (e) { console.warn("Chart destroy failed:", e); }
  if (!canvas) return null;
  try {
    return new Chart(canvas, config);
  } catch (e) {
    console.error("Chart creation failed for", canvas.id, e);
    const errEl = document.getElementById("error");
    if (errEl) errEl.textContent = `Chart "${canvas.id}" failed to render: ${e.message}`;
    return null;
  }
}

// Shared Chart.js tooltip style — light cream card with terracotta title
function tooltipStyle(labelFn) {
  return {
    backgroundColor: PAPER,
    titleColor: ACCENT,
    bodyColor: INK,
    titleFont: { family: '"Newsreader", serif', size: 12, weight: "500", style: "italic" },
    bodyFont:  { family: '"Newsreader", serif', size: 13 },
    titleAlign: "left",
    borderColor: RULE,
    borderWidth: 1,
    padding: { top: 8, right: 12, bottom: 8, left: 12 },
    cornerRadius: 0,
    displayColors: false,
    callbacks: { label: labelFn },
  };
}

function recompute() {
  try {
    const cfg = readConfig();
    const result = estimate(cfg);
    renderResult(cfg, result);
    $("error").textContent = "";
  } catch (e) {
    console.error(e);
    $("error").textContent = "Estimation error: " + e.message;
  }
}

function init() {
  // Populate model preset dropdown, grouped by family.
  const sel = $("model-preset");
  const byFamily = {};
  for (const [name, m] of Object.entries(window.MODEL_PRESETS)) {
    const f = m.family || "Other";
    (byFamily[f] = byFamily[f] || []).push(name);
  }
  const familyOrder = window.MODEL_FAMILY_ORDER || Object.keys(byFamily).sort();
  for (const fam of familyOrder) {
    if (!byFamily[fam] || !byFamily[fam].length) continue;
    const og = document.createElement("optgroup");
    og.label = fam;
    for (const name of byFamily[fam]) {
      const opt = document.createElement("option");
      opt.value = name; opt.textContent = name;
      og.appendChild(opt);
    }
    sel.appendChild(og);
  }
  const customOpt = document.createElement("option");
  customOpt.value = "custom"; customOpt.textContent = "Custom…";
  sel.appendChild(customOpt);
  sel.value = "Llama-3.1-8B";

  // Populate GPU dropdown
  const gpuSel = $("gpu");
  for (const [k, g] of Object.entries(window.GPU_PRESETS)) {
    const opt = document.createElement("option");
    opt.value = k; opt.textContent = g.name;
    gpuSel.appendChild(opt);
  }
  gpuSel.value = "H100 80GB";

  // Wire up listeners
  document.querySelectorAll("input, select").forEach(el => {
    el.addEventListener("input", () => {
      if (el.id === "model-preset") syncPresetFields();
      if (el.id === "strategy") syncStrategyFields();
      recompute();
    });
    el.addEventListener("change", () => {
      if (el.id === "model-preset") syncPresetFields();
      if (el.id === "strategy") syncStrategyFields();
      recompute();
    });
  });

  syncPresetFields();
  syncStrategyFields();
  recompute();
}

window.addEventListener("DOMContentLoaded", init);
