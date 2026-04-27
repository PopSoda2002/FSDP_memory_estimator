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

  // RMSNorm: 2 per layer (input + post-attn) — negligible but we count
  const norms = 2 * h;

  // MLP — dense (SwiGLU) or MoE (num_experts × SwiGLU + router)
  let mlp = 0, expert = 0, router = 0, moe_i = 0, num_experts = 0, top_k = 0;
  if (m.is_moe) {
    moe_i       = m.moe_intermediate_size;
    num_experts = m.num_experts;
    top_k       = m.num_experts_per_tok;
    expert = num_experts * (2 * h * moe_i + moe_i * h);  // gate+up+down per expert
    router = h * num_experts;
  } else {
    mlp = 2 * h * i + i * h;                              // gate+up+down
  }

  // dense_per_layer: weights sharded only by TP (and FSDP within DP)
  // expert_per_layer: also sharded by EP
  const dense_per_layer  = attn + mlp + norms + router;
  const expert_per_layer = expert;
  const per_layer        = dense_per_layer + expert_per_layer;

  const embed = v * h;
  const lm_head = m.tie_embeddings ? 0 : v * h;
  const final_norm = h;

  const total = L * per_layer + embed + lm_head + final_norm;
  return { total, per_layer, dense_per_layer, expert_per_layer,
           embed, lm_head, final_norm, attn, mlp, norms, expert, router,
           q_proj, k_proj, v_proj, o_proj, head_dim, num_q, num_kv,
           is_moe: !!m.is_moe, moe_i, num_experts, top_k };
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

  const N  = cfg.world_size;
  const tp = cfg.tp;                                // tensor parallel (Megatron-style, SP assumed)
  const cp = cfg.cp;                                // context (sequence) parallel
  // Data-parallel size = remaining GPUs after TP × CP. Clamp to ≥1.
  const dp = Math.max(1, Math.floor(N / (tp * cp)));
  // EP only applies to MoE models. Constrained to divide DP cleanly.
  const ep = m.is_moe ? Math.max(1, Math.min(cfg.ep || 1, dp)) : 1;

  const compute_bytes  = PRECISION_BYTES[cfg.compute_dtype];
  const grad_bytes_lp  = compute_bytes;             // gradients in low-prec (cast on reduce)
  const master_bytes   = 4;                         // fp32 master copy held by optimizer
  const optim_bytes    = OPTIM_STATE_BYTES[cfg.optimizer];

  // FSDP sharding applied within the DP dimension.
  // FULL_SHARD       — params, grads, optim all sharded across DP ranks
  // SHARD_GRAD_OP    — params replicated, grads + optim sharded across DP
  // HYBRID_SHARD     — sharded within shard_size, replicated across DP/shard_size
  // NO_SHARD / DDP   — everything replicated within DP
  let shard_param = 1, shard_grad = 1, shard_optim = 1;
  if (cfg.strategy === "FULL_SHARD") {
    shard_param = dp; shard_grad = dp; shard_optim = dp;
  } else if (cfg.strategy === "SHARD_GRAD_OP") {
    shard_param = 1;  shard_grad = dp; shard_optim = dp;
  } else if (cfg.strategy === "HYBRID_SHARD") {
    const s = Math.min(cfg.shard_size || dp, dp);
    shard_param = s;  shard_grad = s;  shard_optim = s;
  }

  // Persistent state, per GPU (bytes).
  // Dense weights (attn, embeds, norms, router) divided by TP × FSDP shard.
  // Expert weights also divided by EP — physically distributed across EP ranks
  // first, then FSDP shards what's left within each EP group. Net divisor =
  // TP × max(EP, shard_factor): under FULL_SHARD with EP ≤ DP this collapses to
  // TP × DP, the same as dense; under DDP/SHARD_GRAD_OP, EP linearly helps.
  const P_expert = m.num_layers * counts.expert_per_layer;  // 0 if dense
  const P_dense  = P - P_expert;
  const denseDiv  = (sf) => tp * sf;
  const expertDiv = (sf) => tp * Math.max(ep, sf);

  const sumByKind = (perParamBytes, sf) =>
    (perParamBytes * P_dense  / denseDiv(sf)) +
    (perParamBytes * P_expert / expertDiv(sf));

  const params_mem = sumByKind(compute_bytes, shard_param);
  const grads_mem  = sumByKind(grad_bytes_lp, shard_grad);
  const master_mem = (cfg.compute_dtype === "fp32") ? 0 : sumByKind(master_bytes, shard_optim);
  const optim_mem  = sumByKind(optim_bytes,   shard_optim);
  const optimizer_total = master_mem + optim_mem;

  // Communication / all-gather buffer (FSDP all-gathers one block at a time;
  // peak = current + prefetched next block, both already TP-sliced; expert
  // portion further reduced by EP since each rank only owns num_experts/EP of them).
  const block_params = (counts.dense_per_layer / tp) + (counts.expert_per_layer / (tp * ep));
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
  const act_dtype_bytes = compute_bytes;

  // Per-layer activation memory (Korthikanti et al. 2022).
  // With Megatron TP+SP, residual stream and MLP intermediates split by TP.
  // With CP, the sequence dimension is split across CP ranks.
  // ⇒ all per-layer activation terms divide by (TP × CP).
  const tcp = tp * cp;
  const act_base = (s * b * h * 17 * act_dtype_bytes / 2) / tcp; // 17 sbh saved tensors
  // MLP intermediates — for MoE, each token routes to top_k experts; with EP,
  // each rank holds num_experts/EP experts, so its share of routed activations
  // is (top_k × s × b) / EP × (gate+up) × moe_intermediate_size.
  const act_mlp = m.is_moe
    ? (s * b * counts.moe_i * counts.top_k * 2 * act_dtype_bytes) / tcp / ep
    : (s * b * i * 2 * act_dtype_bytes) / tcp;
  // Attention scores (no FlashAttention): TP splits heads, CP splits the queries.
  const act_attn = cfg.flash_attn ? 0 : (s * s * b * a * act_dtype_bytes) / tcp;
  const act_per_layer = act_base + act_mlp + act_attn;

  let act_mem = 0;
  if (cfg.act_ckpt === "none") {
    act_mem = L * act_per_layer;
  } else if (cfg.act_ckpt === "selective") {
    act_mem = L * act_per_layer * 0.6;
  } else if (cfg.act_ckpt === "full") {
    // residual stream still split by TP+CP
    act_mem = L * (s * b * h * act_dtype_bytes) / tcp + act_per_layer;
  }

  // Logits + loss: vocab parallel splits V across TP; CP splits sequence.
  const logits_mem = (s * b * v * act_dtype_bytes) / tcp;
  const loss_mem   = (s * b * v * 4) / tcp;
  let head_act_mem = logits_mem + loss_mem;
  if (cfg.ckpt_lm_head) head_act_mem = logits_mem;

  const act_mem_layers = act_mem;
  act_mem += head_act_mem;

  // Embedding output: same TP+CP split as the residual stream.
  const embed_act = (s * b * h * act_dtype_bytes) / tcp;
  act_mem += embed_act;

  // ---------------- Framework / misc ----------------
  const overhead = 1.5 * GB + 0.5 * GB;

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
    _detail: {
      counts, compute_bytes, grad_bytes_lp, master_bytes, optim_bytes,
      tp, cp, dp, ep,
      P_dense, P_expert,
      shard_param, shard_grad, shard_optim,
      params_mem, grads_mem, master_mem, optim_mem,
      block_params, comm_mem,
      s, b, h, a, L, i, v, act_dtype_bytes,
      act_base, act_mlp, act_attn, act_per_layer,
      act_mem_layers, logits_mem, loss_mem, head_act_mem, embed_act,
      overhead,
    },
  };
}

// ---------------------------------------------------------------------------
// UI wiring
// ---------------------------------------------------------------------------

const $  = (id) => document.getElementById(id);
const fmtGB = (bytes) => (bytes / GB).toFixed(2);
const fmtInt = (n) => Math.round(n).toLocaleString();
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
  const { tp, cp, dp, ep } = result._detail;
  const c = result._detail.counts;
  const parallel = c.is_moe
    ? `TP=<strong>${tp}</strong>, CP=<strong>${cp}</strong>, EP=<strong>${ep}</strong>, DP=<strong>${dp}</strong>`
    : `TP=<strong>${tp}</strong>, CP=<strong>${cp}</strong>, DP=<strong>${dp}</strong>`;
  const moeNote = c.is_moe
    ? ` MoE: <strong>${c.num_experts}</strong> experts, <strong>${c.top_k}</strong> active per token.`
    : ``;
  return (
    `Training <strong>${modelLabel}</strong> on <strong>${cfg.world_size}× ${gpu}</strong> ` +
    `(${parallel}) with <strong>${strat}</strong>, <strong>${cfg.compute_dtype}</strong> and <strong>${opt}</strong>, ` +
    `at micro-batch <strong>${cfg.micro_batch}</strong> × seq <strong>${cfg.seq_len}</strong>, ${ckpt}, ${fa}.${moeNote}`
  );
}

// Read a numeric input, falling back to `min` when the field is empty / NaN
// (which happens transiently while the user is editing). Without this, an empty
// world_size would briefly evaluate as 0 and divide-by-zero would push every
// chart dataset to Infinity → Chart.js silently renders nothing.
function numInput(id, min) {
  const v = +$(id).value;
  return Number.isFinite(v) && v >= min ? v : min;
}

function readConfig() {
  const presetName = $("model-preset").value;
  let model;
  if (presetName === "custom") {
    model = {
      hidden_size:       numInput("hidden_size", 64),
      intermediate_size: numInput("intermediate_size", 64),
      num_layers:        numInput("num_layers", 1),
      num_heads:         numInput("num_heads", 1),
      num_kv_heads:      numInput("num_kv_heads", 1),
      vocab_size:        numInput("vocab_size", 1),
      tie_embeddings:    $("tie_embeddings").checked,
    };
  } else {
    model = window.MODEL_PRESETS[presetName];
  }
  return {
    model,
    world_size:    numInput("world_size", 1),
    tp:            numInput("tp", 1),
    cp:            numInput("cp", 1),
    ep:            numInput("ep", 1),
    micro_batch:   numInput("micro_batch", 1),
    seq_len:       numInput("seq_len", 1) * 1024,   // input is in k tokens
    compute_dtype: $("compute_dtype").value,
    optimizer:    $("optimizer").value,
    strategy:     $("strategy").value,
    shard_size:   numInput("shard_size", 1),
    act_ckpt:     $("act_ckpt").value,
    flash_attn:   $("flash_attn").checked,
    ckpt_lm_head: $("ckpt_lm_head").checked,
    gpu:          $("gpu").value,
  };
}

// Show DP = world_size / (TP × CP). Mark invalid when TP × CP doesn't divide
// cleanly, or when EP doesn't divide DP (for MoE models).
function syncDerivedDP() {
  const N  = numInput("world_size", 1);
  const tp = numInput("tp", 1);
  const cp = numInput("cp", 1);
  const ep = numInput("ep", 1);
  const out = $("dp_derived");
  if (!out) return;
  const dp = N / (tp * cp);
  const dpClean = dp >= 1 && Number.isInteger(dp);
  const epRowVisible = $("ep-row").style.display !== "none";
  const epClean = !epRowVisible || (ep <= dp && Number.isInteger(dp / ep));
  if (dpClean && epClean) {
    out.textContent = String(dp);
    out.classList.remove("invalid");
  } else {
    out.textContent = dpClean ? String(dp) : (dp >= 1 ? dp.toFixed(2) : "<1");
    out.classList.add("invalid");
  }
}

function syncPresetFields() {
  const presetName = $("model-preset").value;
  const customFields = $("custom-fields");
  const epRow = $("ep-row");
  if (presetName === "custom") {
    customFields.style.display = "";
    epRow.style.display = "none";
    $("ep").value = 1;
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

  // EP is only meaningful for MoE; reset to 1 when switching away.
  if (m.is_moe) {
    epRow.style.display = "";
  } else {
    epRow.style.display = "none";
    $("ep").value = 1;
  }
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

  // ── Slim horizontal stacked bar (single row, no legend, no axes) ──
  const breakdownDatasets = labels.map((l, idx) => ({
    label: l,
    data: [data[idx]],
    backgroundColor: colors[idx],
    borderColor: PAPER,
    borderWidth: 1,
    borderSkipped: false,
    barPercentage: 1,
    categoryPercentage: 1,
  }));
  if (breakdownChart) {
    breakdownChart.data.datasets = breakdownDatasets;
    breakdownChart.update("none");
  } else {
    breakdownChart = safeChart(null, $("chart-breakdown"), {
      type: "bar",
      data: { labels: [""], datasets: breakdownDatasets },
      options: BREAKDOWN_OPTIONS,
    });
  }

  // ── Doughnut, monochrome-ish ──
  if (pieChart) {
    pieChart.data.labels = labels;
    pieChart.data.datasets[0].data = data;
    pieChart.data.datasets[0].backgroundColor = colors;
    pieChart.update("none");
  } else {
    pieChart = safeChart(null, $("chart-pie"), {
      type: "doughnut",
      data: {
        labels,
        datasets: [{
          data, backgroundColor: colors,
          borderColor: PAPER, borderWidth: 2,
          hoverBorderColor: PAPER, hoverOffset: 4,
        }],
      },
      options: PIE_OPTIONS,
    });
  }

  // ── Sweep curve — single thin terracotta line, dashed reference ──
  const sizes = [1, 2, 4, 8, 16, 32, 64];
  const sweepData = sizes.map(N => {
    try { return estimate({ ...cfg, world_size: N }).total / GB; }
    catch (_) { return null; } // tolerate transient errors so sweep still draws
  });
  const sweepDatasets = [
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
  ];
  if (sweepChart) {
    sweepChart.data.labels = sizes;
    sweepChart.data.datasets = sweepDatasets;
    sweepChart.update();
  } else {
    sweepChart = safeChart(null, $("chart-sweep"), {
      type: "line",
      data: { labels: sizes, datasets: sweepDatasets },
      options: SWEEP_OPTIONS,
    });
  }

  renderDetails(cfg, result);
}

// Static chart options — built once, never mutated. Keeping options stable
// lets us update only `data` on each recompute, which avoids destroy/recreate
// cycles that can leave Chart.js in a "created but not drawn" state.
const AXIS_TICK_FONT = { size: 11, family: '"Newsreader", serif', style: "italic" };
const AXIS_TITLE_FONT = { size: 12, family: '"Newsreader", serif', style: "italic" };
const AXIS_X = {
  grid:   { color: RULE_SOFT, drawTicks: false },
  ticks:  { color: INK_MUTE, font: AXIS_TICK_FONT, padding: 6 },
  border: { color: RULE },
};
const AXIS_Y = {
  grid:   { color: RULE_SOFT, drawTicks: false },
  ticks:  { color: INK_MUTE, font: AXIS_TICK_FONT, padding: 8 },
  border: { color: RULE },
};
const BREAKDOWN_OPTIONS = {
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
};
const PIE_OPTIONS = {
  responsive: true,
  maintainAspectRatio: false,
  cutout: "65%",
  plugins: {
    tooltip: tooltipStyle((ctx) => `${ctx.label} — ${ctx.parsed.toFixed(2)} GB`),
    legend:  { display: false },
  },
};
const SWEEP_OPTIONS = {
  responsive: true,
  maintainAspectRatio: false,
  layout: { padding: { top: 6, right: 4, bottom: 0 } },
  interaction: { mode: "index", intersect: false },
  scales: {
    x: { ...AXIS_X, title: { display: true, text: "world size", color: INK_MUTE, font: AXIS_TITLE_FONT, padding: { top: 8 } } },
    y: { ...AXIS_Y, beginAtZero: true, title: { display: true, text: "GB per GPU", color: INK_MUTE, font: AXIS_TITLE_FONT } },
  },
  plugins: {
    tooltip: tooltipStyle((ctx) => `${ctx.parsed.y.toFixed(2)} GB at ${ctx.parsed.x} GPUs`),
    legend:  {
      display: true, position: "bottom", align: "start",
      labels: {
        color: INK_MUTE,
        font: { family: '"Newsreader", serif', size: 12, style: "italic" },
        boxWidth: 16, boxHeight: 8, padding: 14, usePointStyle: false,
      },
    },
  },
};

// ---------------------------------------------------------------------------
// Detailed calculation breakdown (collapsible)
// ---------------------------------------------------------------------------

function renderDetails(cfg, result) {
  const d = result._detail;
  const c = d.counts;
  const m = cfg.model;

  const sections = [];

  // 1. Notation
  const notationLines = [
    `P   \u2014 total number of model parameters (computed below)`,
    `N   \u2014 world_size, total GPUs                  = ${cfg.world_size}`,
    `TP  \u2014 tensor parallel size                    = ${d.tp}`,
    `CP  \u2014 context (sequence) parallel size        = ${d.cp}`,
    `DP  \u2014 data parallel size = N / (TP*CP)        = ${d.dp}`,
  ];
  if (c.is_moe) {
    notationLines.push(
      `EP  \u2014 expert parallel size (MoE only)         = ${d.ep}    (must divide DP)`,
    );
  }
  notationLines.push(
    `h   \u2014 hidden_size, model hidden dimension     = ${fmtInt(m.hidden_size)}`,
    c.is_moe
      ? `i_e \u2014 moe_intermediate_size (per expert)      = ${fmtInt(c.moe_i)}`
      : `i   \u2014 intermediate_size, MLP hidden dimension  = ${fmtInt(m.intermediate_size)}`,
    `L   \u2014 num_layers, number of transformer layers = ${m.num_layers}`,
    `a   \u2014 num_heads, number of attention heads     = ${m.num_heads}`,
    `s   \u2014 seq_len, sequence length                 = ${fmtInt(cfg.seq_len)}`,
    `b   \u2014 micro_batch, per-GPU batch size          = ${cfg.micro_batch}`,
    `V   \u2014 vocab_size                               = ${fmtInt(m.vocab_size)}`,
  );
  if (c.is_moe) {
    notationLines.push(
      `E   \u2014 num_experts                              = ${c.num_experts}`,
      `k   \u2014 num_experts_per_tok (top-k routing)      = ${c.top_k}`,
    );
  }
  notationLines.push(
    ``,
    `compute_dtype = ${cfg.compute_dtype} \u2192 ${d.compute_bytes} bytes per element`,
    ``,
    `Parallelism convention:`,
    `  TP slices each weight matrix and (with SP) the residual stream by TP.`,
    `  CP slices the sequence dimension by CP across ring-attention ranks.`,
    `  FSDP shards within the DP dimension only.`,
  );
  if (c.is_moe) {
    notationLines.push(
      `  EP shards experts across EP ranks within DP (each rank holds E/EP experts).`,
    );
  }
  sections.push({ title: 'Notation', calc: notationLines.join('\n') });

  // 2. Parameter Count
  const paramLines = [
    `head_dim = h / a = ${m.hidden_size} / ${c.num_q} = ${c.head_dim}`,
    `num_kv_heads = ${c.num_kv}  (for GQA/MQA; equals a if standard MHA)`,
    ``,
    `Attention (per layer):`,
    `  q_proj = h * a * head_dim           = ${m.hidden_size} * ${c.num_q} * ${c.head_dim} = ${fmtInt(c.q_proj)}`,
    `  k_proj = h * num_kv_heads * head_dim = ${m.hidden_size} * ${c.num_kv} * ${c.head_dim} = ${fmtInt(c.k_proj)}`,
    `  v_proj = h * num_kv_heads * head_dim = ${m.hidden_size} * ${c.num_kv} * ${c.head_dim} = ${fmtInt(c.v_proj)}`,
    `  o_proj = a * head_dim * h           = ${c.num_q} * ${c.head_dim} * ${m.hidden_size} = ${fmtInt(c.o_proj)}`,
    `  attn   = q + k + v + o = ${fmtInt(c.attn)}`,
    ``,
  ];
  if (c.is_moe) {
    paramLines.push(
      `MoE block (per layer): E=${c.num_experts} experts \u00d7 SwiGLU(h, i_e) + router`,
      `  per-expert = 3 * h * i_e = 3 * ${m.hidden_size} * ${fmtInt(c.moe_i)} = ${fmtInt(3 * m.hidden_size * c.moe_i)}`,
      `  experts   = E * per-expert = ${c.num_experts} * ${fmtInt(3 * m.hidden_size * c.moe_i)} = ${fmtInt(c.expert)}`,
      `  router    = h * E = ${m.hidden_size} * ${c.num_experts} = ${fmtInt(c.router)}`,
      ``,
      `RMSNorm (per layer): 2 * h = ${fmtInt(c.norms)}`,
      `Per-layer total = attn + experts + router + norms = ${fmtInt(c.per_layer)}`,
      `  dense_per_layer  (sharded by TP only) = ${fmtInt(c.dense_per_layer)}`,
      `  expert_per_layer (sharded by TP \u00d7 EP) = ${fmtInt(c.expert_per_layer)}`,
      ``,
    );
  } else {
    paramLines.push(
      `MLP \u2014 SwiGLU (per layer):`,
      `  gate + up = 2 * h * i = 2 * ${m.hidden_size} * ${fmtInt(m.intermediate_size)} = ${fmtInt(2 * m.hidden_size * m.intermediate_size)}`,
      `  down     = i * h = ${fmtInt(m.intermediate_size)} * ${m.hidden_size} = ${fmtInt(m.intermediate_size * m.hidden_size)}`,
      `  mlp      = ${fmtInt(c.mlp)}`,
      ``,
      `RMSNorm (per layer): 2 * h = 2 * ${m.hidden_size} = ${fmtInt(c.norms)}`,
      `Per-layer total = attn + mlp + norms = ${fmtInt(c.per_layer)}`,
      ``,
    );
  }
  paramLines.push(
    `Embedding  = V * h = ${fmtInt(d.v)} * ${m.hidden_size} = ${fmtInt(c.embed)}`,
    m.tie_embeddings
      ? `LM head    = 0 (tied with embedding)`
      : `LM head    = V * h = ${fmtInt(d.v)} * ${m.hidden_size} = ${fmtInt(c.lm_head)}`,
    `Final norm = h = ${m.hidden_size}`,
    ``,
    `P = L * per_layer + embedding + lm_head + final_norm`,
    `  = ${d.L} * ${fmtInt(c.per_layer)} + ${fmtInt(c.embed)} + ${fmtInt(c.lm_head)} + ${c.final_norm}`,
    `  = ${fmtInt(result.P)}  (${fmtParams(result.P)})`,
  );
  if (c.is_moe) {
    const active = result.P - d.P_expert + d.P_expert * c.top_k / c.num_experts;
    paramLines.push(
      ``,
      `Active params per token = P_dense + P_expert \u00d7 k/E`,
      `                        = ${fmtInt(d.P_dense)} + ${fmtInt(d.P_expert)} \u00d7 ${c.top_k}/${c.num_experts}`,
      `                        \u2248 ${fmtInt(active)}  (${fmtParams(active)})`,
    );
  }
  sections.push({ title: 'Parameter count (P)', calc: paramLines.join('\n') });

  // 3. Model states (sharding + params + grads + optimizer)
  const stratDesc = {
    FULL_SHARD:    'FULL_SHARD (ZeRO-3) \u2014 params, grads, optim all sharded across DP ranks',
    SHARD_GRAD_OP: 'SHARD_GRAD_OP (ZeRO-2) \u2014 params replicated within DP; grads + optim sharded',
    HYBRID_SHARD:  `HYBRID_SHARD \u2014 sharded within shard_size=${cfg.shard_size || d.dp}, replicated across DP/shard_size`,
    NO_SHARD:      'NO_SHARD (DDP) \u2014 fully replicated within DP',
  };
  const optimDesc = {
    adamw:        'AdamW \u2014 1st moment (m) + 2nd moment (v), each fp32 \u2192 8 bytes/param',
    adam:         'Adam \u2014 1st moment (m) + 2nd moment (v), each fp32 \u2192 8 bytes/param',
    sgd_momentum: 'SGD + momentum buffer \u2192 4 bytes/param',
    sgd:          'SGD (vanilla, no state) \u2192 0 bytes/param',
    adam_8bit:    'Adam 8-bit (bnb quantised m+v) \u2192 2 bytes/param',
  };
  const expertNote = c.is_moe
    ? `  Expert weights add an extra divisor: total / (TP \u00d7 max(EP, shard_factor))\n  Dense weights (attn, embed, norm, router): total / (TP \u00d7 shard_factor)`
    : null;
  const fmtSplitFormula = (perBytes, sf, val) => {
    if (!c.is_moe) {
      return [
        `= ${perBytes} * P / TP / shard = ${perBytes} * ${fmtInt(result.P)} / ${d.tp} / ${sf}`,
        `= ${fmtInt(val)} bytes  =  ${fmtGB(val)} GB`,
      ];
    }
    const denseTerm  = perBytes * d.P_dense  / (d.tp * sf);
    const expertTerm = perBytes * d.P_expert / (d.tp * Math.max(d.ep, sf));
    return [
      `= [${perBytes} * P_dense  / (TP \u00d7 shard)] + [${perBytes} * P_expert / (TP \u00d7 max(EP, shard))]`,
      `= [${perBytes} * ${fmtInt(d.P_dense)}  / (${d.tp} \u00d7 ${sf})] + [${perBytes} * ${fmtInt(d.P_expert)} / (${d.tp} \u00d7 max(${d.ep}, ${sf}))]`,
      `= ${fmtGB(denseTerm)} GB + ${fmtGB(expertTerm)} GB = ${fmtGB(val)} GB`,
    ];
  };
  const masterLines = cfg.compute_dtype === 'fp32'
    ? [`  compute_dtype = fp32 \u2192 no master copy needed \u2192 0 bytes`]
    : [`  master_mem ${fmtSplitFormula(4, d.shard_optim, d.master_mem)[0]}`,
       `             ${fmtSplitFormula(4, d.shard_optim, d.master_mem).slice(1).join('\n             ')}`];
  const layoutLine = c.is_moe
    ? `N = ${cfg.world_size}    TP = ${d.tp}    CP = ${d.cp}    EP = ${d.ep}    DP = ${d.dp}`
    : `N = ${cfg.world_size}    TP = ${d.tp}    CP = ${d.cp}    DP = ${d.dp}`;
  const stateLines = [
    `--- Parallelism layout ---`,
    layoutLine,
    ``,
    `--- FSDP sharding (within DP) ---`,
    `${stratDesc[cfg.strategy]}`,
    `shard_param = ${d.shard_param}    shard_grad = ${d.shard_grad}    shard_optim = ${d.shard_optim}`,
    `  (per-GPU memory = total / TP / shard_factor)`,
  ];
  if (expertNote) stateLines.push(expertNote);
  stateLines.push(
    ``,
    `--- Parameters (stored in ${cfg.compute_dtype}, ${d.compute_bytes} bytes/param) ---`,
    `params_mem ${fmtSplitFormula(d.compute_bytes, d.shard_param, d.params_mem)[0]}`,
    `             ${fmtSplitFormula(d.compute_bytes, d.shard_param, d.params_mem).slice(1).join('\n             ')}`,
    ``,
    `--- Gradients (stored in ${cfg.compute_dtype}) ---`,
    `grads_mem  ${fmtSplitFormula(d.grad_bytes_lp, d.shard_grad, d.grads_mem)[0]}`,
    `             ${fmtSplitFormula(d.grad_bytes_lp, d.shard_grad, d.grads_mem).slice(1).join('\n             ')}`,
    ``,
    `--- Optimizer states ---`,
    `Optimizer: ${optimDesc[cfg.optimizer]}`,
    ``,
    `Master weights (fp32 copy, needed when compute_dtype != fp32):`,
    ...masterLines,
    ``,
    `Optimizer state buffers (${d.optim_bytes} bytes per param):`,
    `  optim_mem ${fmtSplitFormula(d.optim_bytes, d.shard_optim, d.optim_mem)[0]}`,
    `              ${fmtSplitFormula(d.optim_bytes, d.shard_optim, d.optim_mem).slice(1).join('\n              ')}`,
    ``,
    `Optimizer total = master_mem + optim_mem = ${fmtGB(d.master_mem)} + ${fmtGB(d.optim_mem)} = ${fmtGB(d.master_mem + d.optim_mem)} GB`,
  );
  sections.push({ title: 'Model states (params + grads + optimizer)', calc: stateLines.join('\n') });

  // 4. Activations
  const tcp = d.tp * d.cp;
  const tcpNote = (tcp > 1)
    ? `  (each per-layer term is then divided by TP*CP = ${d.tp}*${d.cp} = ${tcp})`
    : `  (TP=1 and CP=1, so no parallelism divisor applied)`;
  const actLines = [
    `Per-layer activation breakdown (stored in ${cfg.compute_dtype}, ${d.act_dtype_bytes} bytes):`,
    tcpNote,
    ``,
    `  Saved tensors (intermediate outputs kept for backward pass):`,
    `    = s*b*h * 17 * dtype_bytes / 2 / (TP*CP)`,
    `    = ${fmtInt(d.s)} * ${d.b} * ${fmtInt(d.h)} * 17 * ${d.act_dtype_bytes} / 2 / ${tcp}`,
    `    = ${fmtInt(d.act_base)} bytes`,
    ``,
  ];
  if (c.is_moe) {
    actLines.push(
      `  MoE MLP intermediates — only top-k experts fire per token, and EP further`,
      `  splits experts across ranks (each rank handles 1/EP of routed activations):`,
      `    = s*b * k * 2*i_e * dtype / (TP*CP) / EP`,
      `    = ${fmtInt(d.s)} * ${d.b} * ${c.top_k} * 2*${fmtInt(c.moe_i)} * ${d.act_dtype_bytes} / ${tcp} / ${d.ep}`,
      `    = ${fmtInt(d.act_mlp)} bytes`,
      ``,
    );
  } else {
    actLines.push(
      `  MLP intermediates (gate + up projection outputs):`,
      `    = s*b*i * 2 * dtype_bytes / (TP*CP)`,
      `    = ${fmtInt(d.s)} * ${d.b} * ${fmtInt(d.i)} * 2 * ${d.act_dtype_bytes} / ${tcp}`,
      `    = ${fmtInt(d.act_mlp)} bytes`,
      ``,
    );
  }
  if (!cfg.flash_attn) {
    actLines.push(
      `  Attention score matrix (s*s per head, materialised without FlashAttention):`,
      `    = s*s*b*a * dtype_bytes`,
      `    = ${fmtInt(d.s)} * ${fmtInt(d.s)} * ${d.b} * ${d.a} * ${d.act_dtype_bytes}`,
      `    = ${fmtInt(d.act_attn)} bytes`,
      ``,
    );
  } else {
    actLines.push(`  Attention scores = 0  (FlashAttention avoids materialising the s*s matrix)`, ``);
  }
  actLines.push(`act_per_layer = saved_tensors + mlp + attn_scores = ${fmtInt(d.act_per_layer)} bytes  =  ${fmtGB(d.act_per_layer)} GB`, ``);

  const ckptLabels = {
    none:      'None \u2014 all L layers\u2019 activations kept in memory',
    selective: 'Selective \u2014 drops attention intermediates, keeps ~60% of full',
    full:      'Full \u2014 only residual inputs retained; recompute 1 layer at a time during backward',
  };
  actLines.push(`Activation checkpointing: ${ckptLabels[cfg.act_ckpt]}`);
  if (cfg.act_ckpt === 'none') {
    actLines.push(`  layers_act = L * act_per_layer = ${d.L} * ${fmtInt(d.act_per_layer)} = ${fmtInt(d.act_mem_layers)} bytes`);
  } else if (cfg.act_ckpt === 'selective') {
    actLines.push(`  layers_act = L * act_per_layer * 0.6 = ${d.L} * ${fmtInt(d.act_per_layer)} * 0.6 = ${fmtInt(d.act_mem_layers)} bytes`);
  } else {
    actLines.push(
      `  layers_act = L * (s*b*h*dtype) [residual inputs] + act_per_layer [1 recomputed layer]`,
      `             = ${d.L} * ${fmtInt(d.s * d.b * d.h * d.act_dtype_bytes)} + ${fmtInt(d.act_per_layer)}`,
      `             = ${fmtInt(d.act_mem_layers)} bytes`,
    );
  }
  actLines.push(
    ``,
    `LM head activations (logits tensor is large when V is big):`,
    `  logits = s*b*V * dtype = ${fmtInt(d.s)} * ${d.b} * ${fmtInt(d.v)} * ${d.act_dtype_bytes} = ${fmtInt(d.logits_mem)} bytes`,
    `  loss   = s*b*V * 4 (kept in fp32 for numerical stability) = ${fmtInt(d.loss_mem)} bytes`,
  );
  if (cfg.ckpt_lm_head) {
    actLines.push(`  lm_head checkpointed \u2192 head_act = logits only = ${fmtInt(d.head_act_mem)} bytes`);
  } else {
    actLines.push(`  head_act = logits + loss = ${fmtInt(d.head_act_mem)} bytes`);
  }
  actLines.push(
    ``,
    `Embedding output: s*b*h * dtype = ${fmtInt(d.s)} * ${d.b} * ${fmtInt(d.h)} * ${d.act_dtype_bytes} = ${fmtInt(d.embed_act)} bytes`,
    ``,
    `Total activations = layers_act + head_act + embedding`,
    `  = ${fmtInt(d.act_mem_layers)} + ${fmtInt(d.head_act_mem)} + ${fmtInt(d.embed_act)}`,
    `  = ${fmtInt(d.act_mem_layers + d.head_act_mem + d.embed_act)} bytes  =  ${fmtGB(d.act_mem_layers + d.head_act_mem + d.embed_act)} GB`,
  );
  sections.push({ title: 'Activations', calc: actLines.join('\n') });

  // 5. Communication buffers & framework overhead
  const miscLines = [];
  if (cfg.strategy === 'NO_SHARD') {
    miscLines.push(`Communication buffers:`, `  Strategy = NO_SHARD (DDP) \u2192 no all-gather of parameters needed`, `  comm_mem = 0`);
  } else {
    const blockFormula = c.is_moe
      ? `block_params = dense_per_layer / TP + expert_per_layer / (TP × EP)\n             = ${fmtInt(c.dense_per_layer)} / ${d.tp} + ${fmtInt(c.expert_per_layer)} / (${d.tp} × ${d.ep}) = ${fmtInt(d.block_params)}`
      : `block_params = per_layer / TP = ${fmtInt(c.per_layer)} / ${d.tp} = ${fmtInt(d.block_params)}`;
    miscLines.push(
      `Communication buffers (all-gather for FSDP parameter reconstruction):`,
      `  FSDP reconstructs weights one transformer block at a time.`,
      `  Peak = 2 blocks in memory (current + prefetched), each already TP-sliced.`,
      `  ${blockFormula}`,
      `  comm_mem = compute_bytes * block_params * 2`,
      `           = ${d.compute_bytes} * ${fmtInt(d.block_params)} * 2`,
      `           = ${fmtInt(d.comm_mem)} bytes  =  ${fmtGB(d.comm_mem)} GB`,
    );
  }
  miscLines.push(
    ``,
    `Framework overhead (fixed estimate):`,
    `  CUDA context + NCCL buffers + cuDNN workspace + allocator fragmentation`,
    `  = 1.50 GB (baseline) + 0.50 GB (workspace) = 2.00 GB`,
  );
  sections.push({ title: 'Communication & framework overhead', calc: miscLines.join('\n') });

  // 6. Total
  const comp = result.components;
  sections.push({
    title: 'Total per-GPU peak memory',
    calc: [
      `Parameters (weights)   ${fmtGB(comp['Parameters']).padStart(8)} GB`,
      `Gradients              ${fmtGB(comp['Gradients']).padStart(8)} GB`,
      `Optimizer states       ${fmtGB(comp['Optimizer states']).padStart(8)} GB`,
      `Activations            ${fmtGB(comp['Activations']).padStart(8)} GB`,
      `Comm buffers           ${fmtGB(comp['Comm buffers']).padStart(8)} GB`,
      `Framework overhead     ${fmtGB(comp['Framework']).padStart(8)} GB`,
      `${'─'.repeat(32)}`,
      `Total                  ${fmtGB(result.total).padStart(8)} GB`,
    ].join('\n'),
  });

  let html = '';
  for (const sec of sections) {
    html += `<div class="detail-section"><h3>${sec.title}</h3><pre class="calc">${sec.calc}</pre></div>`;
  }
  $("calc-details-body").innerHTML = html;
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
  sel.value = "Qwen3-8B";

  // Populate GPU dropdown
  const gpuSel = $("gpu");
  for (const [k, g] of Object.entries(window.GPU_PRESETS)) {
    const opt = document.createElement("option");
    opt.value = k; opt.textContent = g.name;
    gpuSel.appendChild(opt);
  }
  gpuSel.value = "H100 80GB";

  // Wire up listeners
  const onChange = (el) => {
    if (el.id === "model-preset") syncPresetFields();
    if (el.id === "strategy") syncStrategyFields();
    if (el.id === "world_size" || el.id === "tp" || el.id === "cp" || el.id === "ep") syncDerivedDP();
    if (el.id === "model-preset") syncDerivedDP();
    recompute();
  };
  document.querySelectorAll("input, select").forEach(el => {
    el.addEventListener("input",  () => onChange(el));
    el.addEventListener("change", () => onChange(el));
  });

  syncPresetFields();
  syncStrategyFields();
  syncDerivedDP();
  recompute();
}

window.addEventListener("DOMContentLoaded", init);
