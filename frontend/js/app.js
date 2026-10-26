(() => {
  "use strict";

  const MODEL_IDS = {
    zeroShot: "chronos_t5_large_zero_shot_adjusted",
    ensemble: "ensemble_chronos_t5_large_gbm_w0.7_2",
    level: "level_calendar_no_model",
  };
  const MODEL_NAMES = {
    [MODEL_IDS.zeroShot]: "Chronos-large zero-shot",
    [MODEL_IDS.ensemble]: "Chronos-large + GBM ensemble",
    [MODEL_IDS.level]: "Level + calendar",
  };
  const STEP_NAMES = [
    "Validate and archive upload",
    "Aggregate sales by date",
    "Merge into full history",
    "Clean the full daily series",
    "Refit calendar and GBM models",
    "Forecast the next 16 days",
  ];
  const COLORS = {
    actual: "#8c9caf",
    newlyAdded: "#28a77f",
    zeroShot: "#8192ad",
    ensemble: "#3269e8",
    level: "#d18b36",
  };

  const state = {
    history: [],
    forecast: [],
    forecastCreatedAt: null,
    range: 90,
    chart: null,
    pollTimer: null,
    health: null,
  };

  const numberFormat = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
  const dayNames = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
  const byId = (id) => document.getElementById(id);

  async function api(path, options) {
    const response = await fetch(path, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = typeof payload.detail === "string" ? payload.detail : `Request failed (${response.status}).`;
      throw new Error(detail);
    }
    return payload;
  }

  function formatDate(isoDate, options = { month: "short", day: "numeric", year: "numeric" }) {
    if (!isoDate) return "—";
    return new Intl.DateTimeFormat("en-US", { ...options, timeZone: "UTC" }).format(new Date(`${isoDate}T00:00:00Z`));
  }

  function weekday(isoDate) {
    return dayNames[new Date(`${isoDate}T00:00:00Z`).getUTCDay()];
  }

  function money(value) {
    return Number.isFinite(value) ? numberFormat.format(value) : "—";
  }

  function setStatus(running, connected = true) {
    const chip = byId("connection-chip");
    chip.className = `status-chip ${connected ? (running ? "status-running" : "") : "status-error"}`;
    chip.innerHTML = `<span class="status-dot"></span>${connected ? (running ? "Job running" : "System ready") : "Connection issue"}`;
    const date = state.health?.last_data_date;
    byId("last-date-chip").textContent = `Latest data ${date ? formatDate(date, { month: "short", day: "numeric" }) : "—"}`;
  }

  function renderSteps(job = null) {
    const list = byId("pipeline-steps");
    const steps = job?.steps || [];
    list.replaceChildren();
    STEP_NAMES.forEach((name, index) => {
      const step = steps[index];
      const status = step?.status || "waiting";
      const row = document.createElement("li");
      row.className = `step-row ${status === "running" ? "is-running" : ""} ${status === "done" ? "is-done" : ""} ${status === "failed" ? "is-failed" : ""}`;
      const marker = document.createElement("span");
      marker.className = "step-marker";
      marker.textContent = status === "done" ? "✓" : status === "failed" ? "!" : String(index + 1);
      const title = document.createElement("span");
      title.textContent = name;
      const time = document.createElement("span");
      time.className = "step-time";
      time.textContent = Number.isFinite(step?.seconds) ? `${step.seconds.toFixed(1)}s` : "";
      row.append(marker, title, time);
      list.append(row);
    });
    byId("job-status-label").textContent = job ? job.status[0].toUpperCase() + job.status.slice(1) : "Ready";
    setStatus(job ? ["queued", "running"].includes(job.status) : false);
  }

  function renderKpis() {
    const last = state.history.at(-1);
    byId("kpi-last-date").textContent = last ? formatDate(last.date, { month: "short", day: "numeric", year: "numeric" }) : "—";
    const ensemble = state.forecast.filter((row) => row.model === MODEL_IDS.ensemble);
    const total = ensemble.reduce((sum, row) => sum + row.forecast, 0);
    byId("kpi-total").textContent = ensemble.length ? money(total) : "—";
    byId("kpi-average").textContent = ensemble.length ? money(total / ensemble.length) : "—";
    const previous = state.history.slice(-16);
    const actualTotal = previous.reduce((sum, row) => sum + row.sales, 0);
    const change = ensemble.length === 16 && actualTotal > 0 ? ((total / actualTotal) - 1) * 100 : null;
    const changeValue = byId("kpi-change");
    changeValue.textContent = change === null ? "—" : `${change >= 0 ? "+" : ""}${change.toFixed(1)}%`;
    changeValue.classList.toggle("kpi-positive", change !== null && change >= 0);
    changeValue.classList.toggle("kpi-negative", change !== null && change < 0);
    byId("kpi-change-caption").textContent = `Ensemble model vs. previous 16 days actual sales`;
  }

  function renderForecastTable() {
    const tbody = byId("forecast-rows");
    tbody.replaceChildren();
    if (!state.forecast.length) {
      const row = document.createElement("tr");
      row.innerHTML = '<td class="table-placeholder" colspan="5">No forecast is available yet.</td>';
      tbody.append(row);
      return;
    }
    const dates = [...new Set(state.forecast.map((row) => row.date))].sort();
    const lookup = new Map(state.forecast.map((row) => [`${row.date}|${row.model}`, row.forecast]));
    dates.forEach((date) => {
      const tr = document.createElement("tr");
      [formatDate(date), weekday(date), ...Object.values(MODEL_IDS).map((model) => money(lookup.get(`${date}|${model}`)))].forEach((value) => {
        const td = document.createElement("td");
        td.textContent = value;
        tr.append(td);
      });
      tbody.append(tr);
    });
    byId("forecast-made-at").textContent = state.forecastCreatedAt
      ? `Forecast made ${new Intl.DateTimeFormat("en-US", { dateStyle: "medium", timeStyle: "short" }).format(new Date(state.forecastCreatedAt))}`
      : "Forecast time unavailable";
  }

  function renderUploads(uploads = []) {
    const tbody = byId("upload-rows");
    tbody.replaceChildren();
    if (!uploads.length) {
      const row = document.createElement("tr");
      row.innerHTML = '<td class="table-placeholder" colspan="4">No uploads yet.</td>';
      tbody.append(row);
      return;
    }
    uploads.slice(0, 8).forEach((upload) => {
      const row = document.createElement("tr");
      const filename = document.createElement("td");
      filename.className = "file-name-cell";
      filename.title = upload.original_file_name || "";
      const createdAt = upload.created_at ? new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }).format(new Date(upload.created_at)) : "—";
      filename.textContent = `${upload.original_file_name || "Upload"} · ${createdAt}`;
      const summary = document.createElement("td");
      summary.textContent = `${numberFormat.format(upload.rows ?? 0)} rows · ${upload.days_added ?? 0} days`;
      const dateRange = document.createElement("td");
      dateRange.textContent = upload.first_date_added && upload.last_date_added
        ? `${formatDate(upload.first_date_added, { month: "short", day: "numeric" })} – ${formatDate(upload.last_date_added, { month: "short", day: "numeric", year: "numeric" })}`
        : "No new dates";
      const statusCell = document.createElement("td");
      const badge = document.createElement("span");
      badge.className = `upload-status ${upload.status}`;
      badge.textContent = upload.status;
      statusCell.append(badge);
      row.append(filename, summary, dateRange, statusCell);
      tbody.append(row);
    });
  }

  function forecastShadePlugin() {
    return {
      id: "forecastShade",
      beforeDatasetsDraw(chart) {
        const firstForecast = state.forecast.map((row) => row.date).sort()[0];
        const lastActual = state.history.at(-1)?.date;
        if (!firstForecast || !lastActual || !chart.chartArea) return;
        const xScale = chart.scales.x;
        const forecastX = xScale.getPixelForValue(firstForecast);
        const previousX = xScale.getPixelForValue(lastActual);
        if (!Number.isFinite(forecastX) || !Number.isFinite(previousX)) return;
        const divider = (forecastX + previousX) / 2;
        const { ctx, chartArea } = chart;
        ctx.save();
        ctx.fillStyle = "rgba(50, 105, 232, 0.055)";
        ctx.fillRect(divider, chartArea.top, chartArea.right - divider, chartArea.bottom - chartArea.top);
        ctx.strokeStyle = "rgba(50, 105, 232, 0.36)";
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(divider, chartArea.top);
        ctx.lineTo(divider, chartArea.bottom);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#6680ad";
        ctx.font = "600 10px Inter, system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("Forecast (next 16 days)", divider + (chartArea.right - divider) / 2, chartArea.top + 13);
        ctx.restore();
      },
    };
  }

  function renderChart() {
    const canvas = byId("sales-chart");
    const history = state.history.slice(-state.range);
    const forecastDates = [...new Set(state.forecast.map((row) => row.date))].sort();
    const labels = [...history.map((row) => row.date), ...forecastDates];
    const historyByDate = new Map(history.map((row) => [row.date, row]));
    const forecastByModel = new Map();
    state.forecast.forEach((row) => {
      if (!forecastByModel.has(row.model)) forecastByModel.set(row.model, new Map());
      forecastByModel.get(row.model).set(row.date, row.forecast);
    });
    const datasets = [
      {
        label: "Actual sales",
        data: labels.map((date) => historyByDate.has(date) && !historyByDate.get(date).is_new ? historyByDate.get(date).sales : null),
        borderColor: COLORS.actual,
        backgroundColor: COLORS.actual,
        borderWidth: 2,
        pointRadius: 0,
        pointHitRadius: 9,
        tension: 0.22,
        spanGaps: false,
      },
    ];
    if (history.some((row) => row.is_new)) {
      datasets.push({
        label: "Newly added data",
        data: labels.map((date) => historyByDate.get(date)?.is_new ? historyByDate.get(date).sales : null),
        borderColor: COLORS.newlyAdded,
        backgroundColor: COLORS.newlyAdded,
        borderWidth: 3,
        pointRadius: 2,
        pointHitRadius: 9,
        tension: 0.22,
        spanGaps: false,
      });
    }
    [
      { id: MODEL_IDS.zeroShot, color: COLORS.zeroShot, dash: [5, 4], width: 1.8 },
      { id: MODEL_IDS.ensemble, color: COLORS.ensemble, dash: [], width: 3.2 },
      { id: MODEL_IDS.level, color: COLORS.level, dash: [8, 4], width: 1.8 },
    ].forEach(({ id, color, dash, width }) => {
      const values = forecastByModel.get(id);
      datasets.push({
        label: MODEL_NAMES[id],
        data: labels.map((date) => values?.has(date) ? values.get(date) : null),
        borderColor: color,
        backgroundColor: color,
        borderWidth: width,
        borderDash: dash,
        pointRadius: 0,
        pointHitRadius: 9,
        tension: 0.2,
        spanGaps: false,
      });
    });
    if (state.chart) state.chart.destroy();
    state.chart = new Chart(canvas, {
      type: "line",
      data: { labels, datasets },
      plugins: [forecastShadePlugin()],
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            position: "bottom",
            labels: { usePointStyle: true, pointStyle: "line", boxWidth: 23, boxHeight: 8, padding: 18, color: "#64748b", font: { size: 10 } },
          },
          tooltip: {
            backgroundColor: "#17263d",
            padding: 11,
            titleFont: { size: 11, weight: "600" },
            bodyFont: { size: 11 },
            callbacks: {
              title(items) {
                const date = items[0]?.label;
                return date ? `${formatDate(date)} · ${weekday(date)}` : "";
              },
              label(context) {
                return `${context.dataset.label}: ${money(context.parsed.y)}`;
              },
            },
          },
        },
        scales: {
          x: {
            type: "category",
            grid: { display: false },
            ticks: { color: "#8996a7", maxTicksLimit: 12, maxRotation: 0, font: { size: 9 } },
            title: { display: true, text: "Date", color: "#78869a", font: { size: 10, weight: "600" } },
          },
          y: {
            beginAtZero: false,
            grid: { color: "rgba(223, 230, 239, 0.7)" },
            ticks: { color: "#8996a7", callback: (value) => numberFormat.format(value), font: { size: 9 } },
            title: { display: true, text: "Total daily sales", color: "#78869a", font: { size: 10, weight: "600" } },
          },
        },
      },
    });
    byId("chart-empty").hidden = labels.length > 0;
  }

  async function refreshDashboard() {
    const [health, history, forecast, uploads] = await Promise.all([
      api("/health"),
      api("/history?days=365"),
      api("/forecast/latest"),
      api("/uploads"),
    ]);
    state.health = health;
    state.history = history;
    state.forecast = forecast.rows || [];
    state.forecastCreatedAt = forecast.created_at;
    setStatus(health.job_running);
    renderKpis();
    renderForecastTable();
    renderUploads(uploads);
    renderChart();
  }

  async function pollJob(jobId) {
    try {
      const job = await api(`/jobs/${encodeURIComponent(jobId)}`);
      renderSteps(job);
      renderUploads(await api("/uploads"));
      if (job.status === "queued" || job.status === "running") {
        state.pollTimer = window.setTimeout(() => pollJob(jobId), 2000);
        return;
      }
      await refreshDashboard();
      renderSteps(job);
      if (job.status === "done") {
        const success = byId("upload-success");
        success.textContent = `Upload processed successfully. ${job.days_added} new day${job.days_added === 1 ? "" : "s"} added; forecast updated.`;
        success.hidden = false;
      } else {
        showUploadError(job.error || "The pipeline job failed.");
      }
    } catch (error) {
      showUploadError(error.message);
      setStatus(false, false);
    }
  }

  function showUploadError(message) {
    const box = byId("upload-error");
    box.textContent = message;
    box.hidden = false;
  }

  async function uploadFile(file) {
    byId("upload-error").hidden = true;
    byId("upload-success").hidden = true;
    if (!file) return;
    if (!/\.(csv|xlsx)$/i.test(file.name)) {
      showUploadError("Upload must be a .xlsx or .csv file.");
      return;
    }
    const form = new FormData();
    form.append("file", file);
    const title = byId("drop-title");
    const previousTitle = title.textContent;
    title.textContent = `Uploading ${file.name}…`;
    try {
      const result = await api("/upload", { method: "POST", body: form });
      renderSteps({ status: "queued", steps: STEP_NAMES.map((name, index) => ({ number: index + 1, name, status: "waiting" })) });
      renderUploads(await api("/uploads"));
      await pollJob(result.job_id);
    } catch (error) {
      showUploadError(error.message);
      try { renderUploads(await api("/uploads")); } catch { /* Keep the upload error visible if activity refresh fails. */ }
    } finally {
      title.textContent = previousTitle;
      byId("file-input").value = "";
    }
  }

  function attachUploadControls() {
    const input = byId("file-input");
    const dropZone = byId("drop-zone");
    input.addEventListener("change", () => uploadFile(input.files[0]));
    ["dragenter", "dragover"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.add("is-dragging");
      });
    });
    ["dragleave", "drop"].forEach((eventName) => {
      dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropZone.classList.remove("is-dragging");
      });
    });
    dropZone.addEventListener("drop", (event) => uploadFile(event.dataTransfer.files[0]));
    document.querySelectorAll(".range-switch button").forEach((button) => {
      button.addEventListener("click", () => {
        document.querySelectorAll(".range-switch button").forEach((item) => {
          item.classList.remove("selected");
          item.setAttribute("aria-pressed", "false");
        });
        button.classList.add("selected");
        button.setAttribute("aria-pressed", "true");
        state.range = Number(button.dataset.days);
        renderChart();
      });
    });
  }

  async function start() {
    attachUploadControls();
    renderSteps();
    renderUploads();
    try {
      await refreshDashboard();
      const recentJob = (await api("/uploads"))[0];
      if (recentJob) {
        const job = await api(`/jobs/${encodeURIComponent(recentJob.job_id)}`);
        renderSteps(job);
        if (["queued", "running"].includes(job.status)) {
          await pollJob(job.job_id);
        } else if (job.status === "done") {
          const success = byId("upload-success");
          success.textContent = `Most recent upload completed successfully. ${job.days_added} new day${job.days_added === 1 ? "" : "s"} added.`;
          success.hidden = false;
        } else if (job.status === "failed") {
          showUploadError(job.error || "The most recent pipeline job failed.");
        }
      }
    } catch (error) {
      setStatus(false, false);
      byId("chart-empty").hidden = false;
      byId("chart-empty").textContent = `Dashboard could not load: ${error.message}`;
      byId("forecast-rows").innerHTML = '<tr><td class="table-placeholder" colspan="5">Forecast unavailable.</td></tr>';
    }
  }

  document.addEventListener("DOMContentLoaded", start);
})();
