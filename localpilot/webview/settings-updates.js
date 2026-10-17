"use strict";

(function () {
  const toggle = document.getElementById("toggle-auto-updates");
  const lastChecked = document.getElementById("update-last-checked");
  const currentVersion = document.getElementById("update-current-version");
  const settingsToggle = document.getElementById("settings-toggle");
  const notice = document.getElementById("update-notice");
  const noticeText = document.getElementById("update-notice-text");
  const apply = document.getElementById("update-notice-apply");
  const later = document.getElementById("update-notice-later");

  if (!toggle || !lastChecked || !currentVersion) return;

  const CHECK_INTERVAL_MS = 30 * 60 * 1000;
  let inFlight = false;
  let cached = null;

  function apiReady() {
    return !!(
      window.pywebview &&
      window.pywebview.api &&
      typeof window.pywebview.api.get_update_settings === "function"
    );
  }

  function formatChecked(value) {
    if (!value) return "Never";
    const date = new Date(value);
    if (!Number.isFinite(date.getTime())) return "Never";
    const now = new Date();
    const sameDay = date.getFullYear() === now.getFullYear()
      && date.getMonth() === now.getMonth()
      && date.getDate() === now.getDate();
    const time = date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    return sameDay ? time : date.toLocaleDateString([], { month: "short", day: "numeric" }) + " " + time;
  }

  function render(status) {
    if (!status || status.ok !== true) return;
    cached = status;
    const enabled = !!status.enabled;
    toggle.classList.toggle("is-on", enabled);
    toggle.setAttribute("aria-checked", String(enabled));
    toggle.disabled = status.canChange === false;
    toggle.title = status.reason || "";
    lastChecked.textContent = status.reason || formatChecked(status.lastChecked);
    currentVersion.textContent = status.currentVersion || "unknown";
    currentVersion.title = status.currentVersion || "unknown";
    if (notice && noticeText && apply && later) {
      notice.hidden = !status.noticeAvailable;
      noticeText.textContent = status.applyRequested
        ? "Update requested. Save any draft and close chat to let Nestra restart safely."
        : "An update is available. Would you like to install it?";
      apply.disabled = !!status.applyRequested;
      later.hidden = !!status.applyRequested;
    }
  }

  function checkDue(status) {
    if (!status || !status.enabled || status.canChange === false) return false;
    if (!status.lastChecked) return true;
    const checked = new Date(status.lastChecked).getTime();
    return !Number.isFinite(checked) || Date.now() - checked >= CHECK_INTERVAL_MS;
  }

  async function loadStatus({ checkIfDue = false } = {}) {
    if (inFlight || !apiReady()) return;
    inFlight = true;
    try {
      let status = await window.pywebview.api.get_update_settings();
      render(status);
      if (checkIfDue && checkDue(status)) {
        status = await window.pywebview.api.check_for_updates();
        render(status);
      }
    } catch (error) {
      console.warn("LocalPilot: update settings unavailable", error);
    } finally {
      inFlight = false;
    }
  }

  toggle.addEventListener("click", async function () {
    if (inFlight || toggle.disabled || !apiReady()) return;
    const wantOn = !toggle.classList.contains("is-on");
    inFlight = true;
    try {
      let status = await window.pywebview.api.set_automatic_updates(wantOn);
      render(status);
      if (wantOn) {
        status = await window.pywebview.api.check_for_updates();
        render(status);
      }
    } catch (error) {
      console.warn("LocalPilot: automatic update preference could not be saved", error);
      if (cached) render(cached);
    } finally {
      inFlight = false;
    }
  });

  async function chooseUpdate(method) {
    if (inFlight || !apiReady() || !cached || !cached.targetSha) return;
    inFlight = true;
    try {
      const result = await window.pywebview.api[method](cached.targetSha);
      render(await window.pywebview.api.get_update_settings());
      if (!result.ok && noticeText) noticeText.textContent = result.reason || "Please check for the update again.";
    } catch (error) {
      if (noticeText) noticeText.textContent = "The update request couldn't be saved. Please try again.";
    } finally { inFlight = false; }
  }
  if (apply) apply.addEventListener("click", function () { chooseUpdate("request_update"); });
  if (later) later.addEventListener("click", function () { chooseUpdate("dismiss_update"); });

  if (settingsToggle) {
    settingsToggle.addEventListener("click", function () {
      setTimeout(function () { loadStatus({ checkIfDue: true }); }, 0);
    });
  }

  window.addEventListener("pywebviewready", function () {
    loadStatus({ checkIfDue: true });
  });

  setInterval(function () {
    if (cached) loadStatus({ checkIfDue: true });
  }, 60 * 1000);
})();
