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
// Telemetry palette — matches CSS custom properties.
const COLORS = {
  "Parameters":       "#ffb547", // amber (dominant)
  "Gradients":        "#5eead4", // teal
  "Optimizer states": "#d6a86a", // tan
  "Activations":      "#f87171", // rose
  "Comm buffers":     "#7ee2b8", // mint
  "Framework":        "#525c70", // muted
};
const INK       = "#dde2ea";
const INK_MUTE  = "#6e7793";
const INK_FAINT = "#404a64";
const RULE      = "#1c2230";
const AMBER     = "#ffb547";
const ROSE      = "#f87171";

// Apply Chart.js global defaults once.
if (window.Chart) {
  Chart.defaults.font.family = '"IBM Plex Mono", ui-monospace, "SF Mono", monospace';
  Chart.defaults.font.size   = 11;
  Chart.defaults.color       = INK_MUTE;
  Chart.defaults.borderColor = RULE;
  Chart.defaults.animation   = { duration: 480, easing: "easeOutQuart" };
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
  // ── KPI banner ──
  $("kpi-params").textContent = fmtParams(result.P);
  $("kpi-total").textContent  = fmtGB(result.total);

  // GPU fit indicator
  const gpu = window.GPU_PRESETS[cfg.gpu];
  const totalGB = result.total / GB;
  const fitEl = $("kpi-fit");
  const barEl = $("fit-bar");
  if (gpu) {
    const pct = (totalGB / gpu.mem_gb) * 100;
    const cls = pct < 70 ? "ok" : pct < 95 ? "warn" : "bad";
    fitEl.textContent = `${pct.toFixed(0)}% · ${gpu.name}`;
    fitEl.className = "kpi-readout fit " + cls;
    barEl.style.width = Math.min(100, pct).toFixed(1) + "%";
    barEl.className = "fit-meter-bar " + cls;
  } else {
    fitEl.textContent = "—";
    fitEl.className = "kpi-readout fit";
    barEl.style.width = "0%";
    barEl.className = "fit-meter-bar";
  }

  // ── Component ledger ──
  const tableBody = $("component-table");
  tableBody.innerHTML = "";
  for (const [k, v] of Object.entries(result.components)) {
    const pct = (v / result.total * 100).toFixed(1);
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><span class="swatch" style="background:${COLORS[k]}"></span>${k}</td>
      <td class="num">${fmtGB(v)}</td>
      <td class="num">${pct}%</td>`;
    tableBody.appendChild(tr);
  }

  // ── Chart palette ──
  const labels = Object.keys(result.components);
  const data   = Object.values(result.components).map(b => b / GB);
  const colors = labels.map(l => COLORS[l]);

  const axisCfg = {
    grid:   { color: RULE, drawTicks: false, lineWidth: 1 },
    ticks:  { color: INK_MUTE, font: { size: 10 }, padding: 8 },
    border: { color: RULE_STRONG_FOR_AXIS },
  };

  // ── Stacked bar (horizontal, full-width readout) ──
  if (breakdownChart) breakdownChart.destroy();
  breakdownChart = new Chart($("chart-breakdown"), {
    type: "bar",
    data: {
      labels: ["▮ PER-GPU"],
      datasets: labels.map((l, idx) => ({
        label: l,
        data: [data[idx]],
        backgroundColor: colors[idx],
        borderColor: "#08090c",
        borderWidth: 1,
        borderSkipped: false,
        barThickness: 40,
      })),
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      layout: { padding: { top: 6, right: 12 } },
      scales: {
        x: {
          stacked: true,
          title:  { display: true, text: "GIGABYTES", color: INK_MUTE, font: { size: 10, weight: "500" } },
          grid:   { color: RULE, drawTicks: false },
          ticks:  { color: INK_MUTE, font: { size: 10 }, padding: 6 },
          border: { color: RULE },
        },
        y: {
          stacked: true,
          grid:   { display: false },
          ticks:  { color: AMBER, font: { size: 11, weight: "600" } },
          border: { color: RULE },
        },
      },
      plugins: {
        tooltip: tooltipStyle((ctx) => `${ctx.dataset.label}  ${ctx.parsed.x.toFixed(2)} GB`),
        legend:  legendStyle(),
      },
    },
  });

  // ── Doughnut ──
  if (pieChart) pieChart.destroy();
  pieChart = new Chart($("chart-pie"), {
    type: "doughnut",
    data: {
      labels,
      datasets: [{
        data,
        backgroundColor: colors,
        borderColor: "#08090c",
        borderWidth: 2,
        hoverBorderColor: "#08090c",
        hoverOffset: 6,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "62%",
      plugins: {
        tooltip: tooltipStyle((ctx) => `${ctx.label}  ${ctx.parsed.toFixed(2)} GB`),
        legend:  legendStyle({ position: "bottom", boxWidth: 8, boxHeight: 8 }),
      },
    },
  });

  // ── Sweep over world_size ──
  const sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
  const sweepData = sizes.map(N => {
    const c = { ...cfg, world_size: N };
    return estimate(c).total / GB;
  });
  // Build vertical-gradient fill for the area under the line
  const sweepCanvas = $("chart-sweep");
  const ctx2d = sweepCanvas.getContext("2d");
  const grad  = ctx2d.createLinearGradient(0, 0, 0, sweepCanvas.height || 320);
  grad.addColorStop(0, "rgba(255, 181, 71, 0.32)");
  grad.addColorStop(1, "rgba(255, 181, 71, 0)");

  if (sweepChart) sweepChart.destroy();
  sweepChart = new Chart(sweepCanvas, {
    type: "line",
    data: {
      labels: sizes,
      datasets: [
        {
          label: "per-gpu peak",
          data: sweepData,
          borderColor: AMBER,
          backgroundColor: grad,
          borderWidth: 1.5,
          fill: true,
          tension: 0.25,
          pointRadius: 3,
          pointBackgroundColor: AMBER,
          pointBorderColor: "#08090c",
          pointBorderWidth: 1,
          pointHoverRadius: 5,
        },
        ...(gpu ? [{
          label: `${gpu.name} ceiling`,
          data: sizes.map(_ => gpu.mem_gb),
          borderColor: ROSE,
          borderDash: [4, 4],
          borderWidth: 1,
          pointRadius: 0,
          fill: false,
        }] : []),
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      layout: { padding: { top: 8, right: 8, bottom: 4 } },
      interaction: { mode: "index", intersect: false },
      scales: {
        x: {
          title:  { display: true, text: "WORLD_SIZE", color: INK_MUTE, font: { size: 10, weight: "500" }, padding: { top: 6 } },
          grid:   { color: RULE, lineWidth: 1, drawTicks: false },
          ticks:  { color: INK_MUTE, font: { size: 10 }, padding: 6 },
          border: { color: RULE },
        },
        y: {
          beginAtZero: true,
          title:  { display: true, text: "GB / GPU", color: INK_MUTE, font: { size: 10, weight: "500" } },
          grid:   { color: RULE, lineWidth: 1, drawTicks: false },
          ticks:  { color: INK_MUTE, font: { size: 10 }, padding: 6 },
          border: { color: RULE },
        },
      },
      plugins: {
        tooltip: tooltipStyle((ctx) =>
          `${ctx.dataset.label}  ${ctx.parsed.y.toFixed(2)} GB @ ${ctx.parsed.x} gpus`),
        legend:  legendStyle({ position: "bottom" }),
      },
    },
  });
}

const RULE_STRONG_FOR_AXIS = "#2e3a52";

// Shared Chart.js style helpers
function tooltipStyle(labelFn) {
  return {
    backgroundColor: "#050608",
    titleColor: AMBER,
    bodyColor: INK,
    titleFont: { family: '"IBM Plex Mono", monospace', size: 10, weight: "600" },
    bodyFont:  { family: '"IBM Plex Mono", monospace', size: 11 },
    titleAlign: "left",
    borderColor: RULE_STRONG_FOR_AXIS,
    borderWidth: 1,
    padding: 10,
    cornerRadius: 0,
    displayColors: true,
    boxWidth: 8,
    boxHeight: 8,
    callbacks: { label: labelFn },
  };
}
function legendStyle(extra = {}) {
  return {
    position: "bottom",
    align: "start",
    labels: {
      color: INK_MUTE,
      font: { family: '"IBM Plex Mono", monospace', size: 10, weight: "500" },
      boxWidth: 10,
      boxHeight: 10,
      padding: 12,
      usePointStyle: false,
    },
    ...extra,
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

// ── Status bar clock + session tag ──
function startStatusBar() {
  const pad = (n) => String(n).padStart(2, "0");
  const rev = $("sb-rev");
  if (rev) {
    // Stable per-session tag derived from time, gives the readout a "build id" feel.
    const t = Date.now().toString(36).slice(-6).toUpperCase();
    rev.textContent = "0x" + t;
  }
  const tick = () => {
    const el = $("sb-clock");
    if (!el) return;
    const d = new Date();
    el.textContent =
      pad(d.getUTCHours()) + ":" + pad(d.getUTCMinutes()) + ":" + pad(d.getUTCSeconds());
  };
  tick();
  setInterval(tick, 1000);
}

function init() {
  // Populate model preset dropdown
  const sel = $("model-preset");
  for (const name of Object.keys(window.MODEL_PRESETS)) {
    const opt = document.createElement("option");
    opt.value = name; opt.textContent = name;
    sel.appendChild(opt);
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
  startStatusBar();
  recompute();
}

window.addEventListener("DOMContentLoaded", init);
