"use strict";
(function () {
  "use strict";

  const VALID_STATES = new Set([
    "idle", "listening", "thinking", "researching", "working", "speaking",
    "success", "uncertain", "error", "learning", "restarting", "sleeping", "offline",
  ]);
  let lastPublished = null;
  let lastSystemOpen = null;

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

  function publishSystemLayout() {
    const panel = document.getElementById("panel");
    if (!panel) return;
    const isOpen = panel.classList.contains("is-system-open");
    if (isOpen === lastSystemOpen) return;
    const bridge = bridgeApi();
    if (!bridge || typeof bridge.set_systemsense_open !== "function") return;
    lastSystemOpen = isOpen;
    Promise.resolve(bridge.set_systemsense_open(isOpen)).catch(function () {
      lastSystemOpen = null;
    });
  }

  const stateObserver = new MutationObserver(function (records) {
    if (records.some(function (record) { return record.attributeName === "data-state"; })) {
      publishCurrentState();
    }
  });
  stateObserver.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-state"],
  });

  const panel = document.getElementById("panel");
  if (panel) {
    const layoutObserver = new MutationObserver(function (records) {
      if (records.some(function (record) { return record.attributeName === "class"; })) {
        publishSystemLayout();
      }
    });
    layoutObserver.observe(panel, { attributes: true, attributeFilter: ["class"] });
  }

  function publishAll() {
    publishCurrentState();
    publishSystemLayout();
  }

  window.addEventListener("pywebviewready", publishAll);
  window.addEventListener("pagehide", function () {
    const bridge = bridgeApi();
    if (!bridge) return;
    if (typeof bridge.clear_companion_state === "function") bridge.clear_companion_state();
    if (typeof bridge.set_systemsense_open === "function") bridge.set_systemsense_open(false);
  });

  publishAll();
})();
