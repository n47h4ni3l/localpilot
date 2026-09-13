"use strict";
(function () {
  "use strict";
  const VALID_STATES = new Set(["idle", "listening", "thinking", "researching", "working", "speaking", "success", "uncertain", "error", "learning", "restarting", "sleeping", "offline"]);
  let lastPublished = null;

  function bridgeApi() {
    return window.pywebview && window.pywebview.api ? window.pywebview.api : null;
  }

  function publishCurrentState() {
    const state = String(document.documentElement.dataset.state || "").toLowerCase();
    if (!VALID_STATES.has(state) || state === lastPublished) return;
    const bridge = bridgeApi();
    if (!bridge || typeof bridge.set_companion_state !== "function") return;
    lastPublished = state;
    Promise.resolve(bridge.set_companion_state(state)).catch(function () {
      lastPublished = null;
    });
  }

  const observer = new MutationObserver(function (records) {
    if (records.some(function (record) { return record.attributeName === "data-state"; })) {
      publishCurrentState();
    }
  });
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-state"] });

  window.addEventListener("pywebviewready", publishCurrentState);
  window.addEventListener("pagehide", function () {
    const bridge = bridgeApi();
    if (bridge && typeof bridge.clear_companion_state === "function") bridge.clear_companion_state();
  });
  publishCurrentState();
})();
