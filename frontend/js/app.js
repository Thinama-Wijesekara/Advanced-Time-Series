(() => {
  "use strict";
  const IDS = { zero: "chronos_t5_large_zero_shot_adjusted", ensemble: "ensemble_chronos_t5_large_gbm_w0.7_2", level: "level_calendar_no_model" };
  const names = { [IDS.zero]: "Chronos-large zero-shot", [IDS.ensemble]: "Chronos-large + GBM ensemble", [IDS.level]: "Level + calendar" };
  const colors = { actual: "#7c91a8", new: "#24a47d", forecast: "#3269e8" };
  let model = localStorage.getItem("selectedModel") || IDS.level;
  if (!Object.values(IDS).includes(model)) model = IDS.level;
  let chart = null, pendingTimer = null, runTimer = null, activeRunId = null;
  const $ = id => document.getElementById(id);
  async function api(path, options = {}) {
    const response = await fetch(path, { cache: "no-store", ...options });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || payload.database_error || `Request failed (${response.status})`);
    return payload;
  }
  const money = value => Number.isFinite(Number(value)) ? new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(value) : "—";
  const dateLabel = value => value ? new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" }).format(new Date(`${value}T00:00:00Z`)) : "—";

  async function refreshHealth() {
    try {
      const health = await api("/health");
      $("connection-chip").className = "status-chip";
      $("connection-chip").textContent = "Connected";
      $("data-chip").hidden = Boolean(health.database_ok);
      if (!health.database_ok) $("data-chip-text").textContent = health.database_error || "Database error";
      $("last-date-chip").textContent = `Latest data ${health.last_data_date ? dateLabel(health.last_data_date) : "—"}`;
    } catch (error) {
      $("connection-chip").className = "status-chip status-error";
      $("connection-chip").textContent = "Connection issue";
    }
  }

  function renderPending(data) {
    const box = $("pending-batches");
    box.replaceChildren();
    const batches = data.batches || [];
    if (!batches.length) box.innerHTML = '<p class="empty-copy">No pending batches. New database rows will appear here.</p>';
    batches.forEach(batch => {
      const div = document.createElement("div"); div.className = "batch-row";
      div.innerHTML = `<strong>${batch.source || "database"}</strong><span>${batch.first_date || "—"} – ${batch.last_date || "—"}</span><span>${batch.rows} rows</span>`;
      box.append(div);
    });
    $("pending-total").textContent = `${data.pending_rows || 0} pending rows`;
    const active = $("start-data").dataset.active === "true";
    $("start-data").disabled = !data.pending_rows || active;
    $("start-help").textContent = active ? "A run is active." : (!data.pending_rows ? "No pending data is available." : "Pending database rows are ready to process.");
    if (data.direct_rows) $("direct-note").textContent = `${data.direct_rows} rows were inserted directly with no batch.`;
  }

  async function refreshPending() {
    try { renderPending(await api("/api/pending")); } catch (error) { $("pending-total").textContent = error.message; }
  }

  const DATA_STEPS = ["Read pending data", "Aggregate by date", "Merge into history", "Clean the series", "Refit calendar and gradient boosting"];
  function renderDataSteps(current, status) {
    const currentIndex = DATA_STEPS.indexOf(current);
    document.querySelectorAll("[data-step]").forEach(row => {
      const index = DATA_STEPS.indexOf(row.dataset.step);
      row.classList.toggle("is-running", status !== "done" && status !== "failed" && index === currentIndex);
      row.classList.toggle("is-done", status === "done" || index < currentIndex);
      row.classList.toggle("is-failed", status === "failed" && index === currentIndex);
      const marker = row.querySelector(".step-marker");
      if (row.classList.contains("is-done")) marker.textContent = "✓";
      else if (row.classList.contains("is-failed")) marker.textContent = "!";
      else marker.textContent = String(index + 1);
    });
  }

  async function pollRun(runId) {
    activeRunId = runId;
    $("start-data").dataset.active = "true";
    if (runTimer) clearInterval(runTimer);
    const poll = async () => {
      try {
        const run = await api(`/api/runs/${runId}`);
        renderDataSteps(run.step, run.status);
        if (run.status === "done") {
          clearInterval(runTimer); runTimer = null; activeRunId = null;
          $("start-data").dataset.active = "false";
          await refreshAll();
          await computeForecast();
        } else if (run.status === "failed") {
          clearInterval(runTimer); runTimer = null; activeRunId = null;
          $("start-data").dataset.active = "false";
          alert(run.error || "Data processing failed.");
          await refreshAll();
        }
      } catch (error) { $("forecast-state").textContent = error.message; }
    };
    await poll();
    runTimer = setInterval(poll, 1000);
  }

  async function startData() {
    try {
      const result = await api("/api/start", { method: "POST" });
      await pollRun(result.run_id);
    } catch (error) { alert(error.message); }
  }

  async function loadHistory() {
    const history = await api("/api/history?days=120");
    renderChart(history.rows || [], stateForecastRows());
  }
  let selectedForecast = [];
  function stateForecastRows() { return selectedForecast; }

  async function loadForecast() {
    try {
      const response = await api(`/api/forecast?model=${encodeURIComponent(model)}`);
      selectedForecast = response.rows || [];
      renderForecastState(response);
      await loadHistory();
    } catch (error) {
      selectedForecast = [];
      $("forecast-state").textContent = error.message;
      await loadHistory();
    }
  }

  function renderForecastState(response) {
    const panel = $("forecast-action"); panel.replaceChildren();
    if (response.state === "ready") {
      const rows = response.rows || []; selectedForecast = rows;
      const total = rows.reduce((sum, row) => sum + Number(row.forecast || 0), 0);
      $("kpi-total").textContent = money(total);
      $("kpi-average").textContent = money(rows.length ? total / rows.length : 0);
      $("forecast-model-title").textContent = names[model];
      renderForecastTable(rows);
      const a = document.createElement("a"); a.className = "button button-secondary"; a.href = `/api/forecast/download?model=${encodeURIComponent(model)}`; a.textContent = "Download CSV"; panel.append(a);
    } else if (model === IDS.level) {
      $("forecast-state").textContent = "Computing Level + calendar forecast…";
      computeForecast();
    } else if (response.state === "computing") {
      $("forecast-state").textContent = `Computing forecast for ${names[model]}…`;
    } else {
      $("forecast-state").textContent = `Forecast not computed for the latest data.`;
      const button = document.createElement("button"); button.className = "button button-primary"; button.textContent = "Compute forecast"; button.onclick = computeForecast; panel.append(button);
    }
  }

  function renderForecastTable(rows) {
    const body = $("forecast-rows"); body.replaceChildren();
    if (!rows.length) { body.innerHTML = '<tr><td class="table-placeholder" colspan="3">No forecast rows.</td></tr>'; return; }
    rows.forEach(row => { const tr = document.createElement("tr"); tr.innerHTML = `<td>${dateLabel(row.date)}</td><td>${new Date(`${row.date}T00:00:00Z`).toLocaleDateString("en-US", { weekday: "long", timeZone: "UTC" })}</td><td>${money(row.forecast)}</td>`; body.append(tr); });
  }

  async function computeForecast() {
    try { await api(`/api/forecast/run?model=${encodeURIComponent(model)}`, { method: "POST" }); await loadForecast(); }
    catch (error) { $("forecast-state").textContent = error.message; }
  }

  function renderChart(history, forecast) {
    const ctx = $("sales-chart"); if (!ctx || !window.Chart) return;
    if (chart) chart.destroy();

    const labels = [...new Set([
      ...history.map(row => row.date),
      ...forecast.map(row => row.date)
    ])].sort();
    const actualByDate = new Map(history.map(row => [row.date, Number(row.sales)]));
    const newByDate = new Map(history.filter(row => row.is_new).map(row => [row.date, Number(row.sales)]));
    const forecastByDate = new Map(forecast.map(row => [row.date, Number(row.forecast)]));
    const valuesFor = map => labels.map(date => map.has(date) ? map.get(date) : null);

    const datasets = [{
      label: "Actual sales",
      data: valuesFor(actualByDate),
      borderColor: colors.actual,
      backgroundColor: "transparent",
      borderWidth: 2,
      pointRadius: 0,
      spanGaps: false
    }];
    if (newByDate.size) datasets.push({
      label: "Newly added",
      data: valuesFor(newByDate),
      borderColor: colors.new,
      backgroundColor: colors.new,
      pointRadius: 4,
      showLine: false
    });
    if (forecastByDate.size) datasets.push({
      label: "Selected forecast",
      data: valuesFor(forecastByDate),
      borderColor: colors.forecast,
      backgroundColor: "transparent",
      borderWidth: 2,
      pointRadius: 2,
      spanGaps: false
    });

    chart = new Chart(ctx, {
      type: "line",
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        scales: { x: { ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 12 } }, y: { beginAtZero: false } },
        plugins: { legend: { position: "bottom" } }
      }
    });
  }

  function selectModel(id) {
    model = id; localStorage.setItem("selectedModel", model); location.hash = model;
    document.querySelectorAll(".model-button").forEach(button => { const selected = button.dataset.model === model; button.classList.toggle("selected", selected); button.setAttribute("aria-pressed", String(selected)); });
    $("forecast-model-title").textContent = names[model]; selectedForecast = []; loadForecast();
  }

  async function refreshBatches() {
    const batches = await api("/api/batches"); const body = $("batch-rows"); body.replaceChildren();
    (batches.rows || []).slice(0, 8).forEach(batch => { const tr = document.createElement("tr"); tr.innerHTML = `<td>${batch.original_file_name || batch.source || "Database insert"}</td><td>${batch.rows}</td><td>${batch.status}</td><td>${batch.message || "—"}</td>`; body.append(tr); });
  }

  async function refreshAll() { await Promise.all([refreshHealth(), refreshPending(), loadHistory(), refreshBatches(), loadForecast()]); }

  window.addEventListener("DOMContentLoaded", () => {
    const hash = location.hash.slice(1); if (Object.values(IDS).includes(hash)) model = hash; localStorage.setItem("selectedModel", model);
    $("start-data").onclick = startData;
    document.querySelectorAll(".model-button").forEach(button => { const selected = button.dataset.model === model; button.classList.toggle("selected", selected); button.setAttribute("aria-pressed", String(selected)); button.onclick = () => selectModel(button.dataset.model); });
    refreshAll(); pendingTimer = setInterval(() => { refreshHealth(); refreshPending(); }, 10000);
  });
})();
