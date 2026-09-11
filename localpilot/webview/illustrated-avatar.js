"use strict";

/* LocalPilot illustrated avatar overlay.
 *
 * The production state store in app.js continues to own runtime state and the
 * original canvas renderer remains active as a fail-safe.  Once sprite.png is
 * available, this layer covers each canvas with the matching hand-drawn pose.
 * Two independently redrawn frames are alternated at a deliberately modest
 * cadence, producing a traditional line-boil / living-sketch effect rather
 * than smooth digital tweening.
 */
(function () {
  "use strict";

  const ROW = {
    idle: 0,
    listening: 1,
    thinking: 2,
    researching: 3,
    working: 4,
    speaking: 1,
    success: 5,
    error: 6,
    uncertain: 7,
    learning: 8,
    restarting: 4,
    sleeping: 0,
    offline: 6,
  };

  const FRAME_HOLD_MS = 360;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const canvases = [
    document.getElementById("avatar-dock"),
    document.getElementById("avatar-header"),
  ].filter(Boolean);

  if (!canvases.length) return;

  class IllustratedAvatarLayer {
    constructor(canvas) {
      this.canvas = canvas;
      this.enabled = false;
      this.overlay = document.createElement("span");
      this.overlay.className = "illustrated-avatar-layer";
      this.overlay.setAttribute("aria-hidden", "true");
      Object.assign(this.overlay.style, {
        position: "absolute",
        display: "block",
        pointerEvents: "none",
        backgroundRepeat: "no-repeat",
        backgroundOrigin: "border-box",
        backgroundClip: "border-box",
        zIndex: "2",
      });

      const parent = canvas.parentElement;
      if (parent && window.getComputedStyle(parent).position === "static") {
        parent.style.position = "relative";
      }
      if (parent) parent.appendChild(this.overlay);
      this.syncGeometry();
    }

    syncGeometry() {
      const canvas = this.canvas;
      const width = canvas.offsetWidth || canvas.width;
      const height = canvas.offsetHeight || canvas.height;
      this.width = width;
      this.height = height;
      this.overlay.style.left = canvas.offsetLeft + "px";
      this.overlay.style.top = canvas.offsetTop + "px";
      this.overlay.style.width = width + "px";
      this.overlay.style.height = height + "px";
      this.overlay.style.backgroundSize = (width * 2) + "px " + (height * 9) + "px";
    }

    enable(spriteUrl) {
      this.overlay.style.backgroundImage = 'url("' + spriteUrl + '")';
      this.canvas.style.opacity = "0";
      this.enabled = true;
      this.syncGeometry();
    }

    draw(state, frame) {
      if (!this.enabled) return;
      const row = Object.prototype.hasOwnProperty.call(ROW, state) ? ROW[state] : ROW.error;
      this.overlay.style.backgroundPosition =
        (-frame * this.width) + "px " + (-row * this.height) + "px";
    }
  }

  const layers = canvases.map(function (canvas) { return new IllustratedAvatarLayer(canvas); });

  function currentState() {
    const state = document.documentElement.dataset.state || "restarting";
    return Object.prototype.hasOwnProperty.call(ROW, state) ? state : "error";
  }

  const observer = new MutationObserver(function (records) {
    if (records.some(function (record) { return record.attributeName === "data-state"; })) {
      const state = currentState();
      layers.forEach(function (layer) { layer.draw(state, 0); });
    }
  });
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-state"] });

  window.addEventListener("resize", function () {
    layers.forEach(function (layer) { layer.syncGeometry(); });
  });

  const sprite = new Image();
  sprite.onload = function () {
    layers.forEach(function (layer) { layer.enable(sprite.src); });
    requestAnimationFrame(animate);
  };
  sprite.onerror = function () {
    // Deliberately do nothing: app.js' existing canvas avatar remains visible.
    console.warn("LocalPilot: illustrated avatar asset unavailable; using pixel fallback.");
  };
  sprite.src = "avatar/sprite.png";

  function animate(now) {
    const frame = reducedMotion ? 0 : (Math.floor(now / FRAME_HOLD_MS) % 2);
    const state = currentState();
    layers.forEach(function (layer) { layer.draw(state, frame); });
    requestAnimationFrame(animate);
  }
})();
