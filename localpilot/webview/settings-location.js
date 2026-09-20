"use strict";

(function () {
  const toggle = document.getElementById("toggle-location");
  const statusEl = document.getElementById("location-status");
  const sourceEl = document.getElementById("location-source");
  const refreshBtn = document.getElementById("location-refresh");
  const settingsToggle = document.getElementById("settings-toggle");

  if (!toggle || !statusEl || !sourceEl || !refreshBtn) return;

  let inFlight = false;
  let cached = null;

  function apiReady() {
    return !!(
      window.pywebview &&
      window.pywebview.api &&
      typeof window.pywebview.api.get_location_settings === "function"
    );
  }

  function formatAccuracy(value) {
    const n = Number(value);
    if (!Number.isFinite(n) || n <= 0) return "";
    if (n < 1000) return " ±" + Math.round(n) + " m";
    return " ±" + (n / 1000).toFixed(1) + " km";
  }

  function render(status) {
    if (!status || status.ok !== true) return;
    cached = status;

    const enabled = !!status.enabled;
    const available = !!status.available;

    toggle.classList.toggle("is-on", enabled);
    toggle.setAttribute("aria-checked", String(enabled));
    refreshBtn.disabled = !enabled || inFlight;

    if (!enabled) {
      statusEl.textContent = "Off";
      sourceEl.textContent = "—";
      statusEl.title = "";
      return;
    }

    if (inFlight) {
      statusEl.textContent = "Reading Windows location…";
    } else if (available) {
      statusEl.textContent = (status.stale ? "Available · stale" : "Available") + formatAccuracy(status.accuracyM);
    } else {
      statusEl.textContent = status.error ? "Unavailable" : "Waiting for location";
    }

    statusEl.title = status.error || "";
    sourceEl.textContent = status.source || "Windows Location Service";
    sourceEl.title = status.source || "";
  }

  async function loadStatus() {
    if (inFlight || !apiReady()) return;
    inFlight = true;
    try {
      const status = await window.pywebview.api.get_location_settings();
      inFlight = false;
      render(status);
    } catch (error) {
      inFlight = false;
      console.warn("LocalPilot: location settings unavailable", error);
      if (cached) render(cached);
    }
  }

  toggle.addEventListener("click", async function () {
    if (inFlight || !apiReady()) return;
    const wantOn = !toggle.classList.contains("is-on");
    inFlight = true;
    render(cached || { ok: true, enabled: wantOn, available: false });
    try {
      const status = await window.pywebview.api.set_location_enabled(wantOn);
      inFlight = false;
      render(status);
    } catch (error) {
      inFlight = false;
      console.warn("LocalPilot: location preference could not be saved", error);
      if (cached) render(cached);
    }
  });

  refreshBtn.addEventListener("click", async function () {
    if (inFlight || !apiReady() || !toggle.classList.contains("is-on")) return;
    inFlight = true;
    render(cached || { ok: true, enabled: true, available: false });
    try {
      const status = await window.pywebview.api.refresh_location();
      inFlight = false;
      render(status);
    } catch (error) {
      inFlight = false;
      console.warn("LocalPilot: location refresh failed", error);
      if (cached) render(cached);
    }
  });

  if (settingsToggle) {
    settingsToggle.addEventListener("click", function () {
      setTimeout(loadStatus, 0);
    });
  }

  window.addEventListener("pywebviewready", loadStatus);
})();
