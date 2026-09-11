"use strict";

/* LocalPilot illustrated avatar overlay.
 *
 * app.js remains the owner of runtime state and its pixel canvas remains the
 * fail-safe.  The illustrated renderer deliberately separates three kinds of
 * motion so the character feels alive without looking electronically jittery:
 *
 *   1. LINE BOIL: the two independent redraws alternate slowly.  This is the
 *      passive hand-drawn wiggle, perceived mostly around outer contours.
 *   2. STATE LOOP: each pose has a small repeating physical action -- breath,
 *      listen, write, type, speak, read, hesitate, etc.
 *   3. TRANSITION: an old pose gently leaves while the new pose fades/settles
 *      in, rather than snapping from one status card to another.
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

  const LINE_BOIL_MS = 520;
  const TRANSITION_MS = 280;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const canvases = [
    document.getElementById("avatar-dock"),
    document.getElementById("avatar-header"),
  ].filter(Boolean);

  if (!canvases.length) return;

  function clamp01(value) { return Math.max(0, Math.min(1, value)); }
  function easeOutCubic(value) {
    const x = 1 - clamp01(value);
    return 1 - x * x * x;
  }
  function wave(seconds, period) {
    return Math.sin((seconds / period) * Math.PI * 2);
  }

  function stateMotion(state, ageMs) {
    if (reducedMotion) return { x: 0, y: 0, scale: 1, rotate: 0, opacity: 1 };
    const t = Math.max(0, ageMs) / 1000;
    let x = 0, y = 0, scale = 1, rotate = 0, opacity = 1;

    switch (state) {
      case "idle":
        // Almost imperceptible breathing. The line boil does the visual work.
        y = 0.45 * wave(t, 3.6);
        scale = 1 + 0.004 * wave(t, 3.6);
        break;
      case "listening":
        // A tiny recurring lean toward the raised hand.
        x = 0.45 + 0.45 * wave(t, 1.8);
        rotate = 0.35 * wave(t, 1.8);
        break;
      case "thinking":
        // Short writing rhythm; the pencil/notepad are already in the artwork.
        x = 0.45 * wave(t, 0.62);
        y = 0.22 * wave(t, 1.24);
        rotate = 0.12 * wave(t, 0.62);
        break;
      case "researching":
        // Slow scan/lean over the book and magnifying glass.
        y = 0.45 * wave(t, 2.1);
        rotate = 0.28 * wave(t, 2.1);
        break;
      case "working":
        // Restrained typing cadence, deliberately smaller than the line width.
        x = 0.28 * wave(t, 0.48);
        y = 0.20 * wave(t, 0.96);
        break;
      case "speaking":
        y = -0.45 * wave(t, 0.8);
        scale = 1 + 0.006 * (0.5 + 0.5 * wave(t, 0.8));
        break;
      case "success": {
        // One quick celebratory lift, then a quiet settled loop.
        if (ageMs < 760) {
          const p = ageMs / 760;
          y = -5.0 * Math.sin(Math.PI * p) * (1 - 0.30 * p);
          rotate = -1.2 * Math.sin(Math.PI * p);
          scale = 1 + 0.035 * Math.sin(Math.PI * p);
        } else {
          y = 0.3 * wave(t, 2.4);
        }
        break;
      }
      case "uncertain":
        x = 0.75 * wave(t, 2.2);
        rotate = 0.65 * wave(t, 2.2);
        break;
      case "error":
        // Brief reaction only; persistent frantic shaking would be irritating.
        if (ageMs < 460) {
          const decay = 1 - ageMs / 460;
          x = 2.4 * decay * Math.sin(ageMs / 24);
          rotate = 0.8 * decay * Math.sin(ageMs / 31);
        }
        break;
      case "restarting":
        y = 0.35 * wave(t, 1.1);
        opacity = 0.82 + 0.18 * (0.5 + 0.5 * wave(t, 1.1));
        break;
      case "sleeping":
        y = 0.4 * wave(t, 5.8);
        scale = 1 + 0.0035 * wave(t, 5.8);
        break;
      case "learning":
        // Gentle notebook/nod loop.
        y = -0.45 * Math.max(0, wave(t, 2.0));
        rotate = -0.24 * Math.max(0, wave(t, 2.0));
        break;
      case "offline":
        opacity = 0.66;
        break;
      default:
        break;
    }
    return { x: x, y: y, scale: scale, rotate: rotate, opacity: opacity };
  }

  function motionTransform(motion) {
    return "translate(" + motion.x.toFixed(2) + "px," + motion.y.toFixed(2) + "px) " +
      "scale(" + motion.scale.toFixed(4) + ") rotate(" + motion.rotate.toFixed(2) + "deg)";
  }

  class IllustratedAvatarLayer {
    constructor(canvas) {
      this.canvas = canvas;
      this.enabled = false;
      this.state = "restarting";
      this.previousState = "restarting";
      this.stateStart = performance.now();
      this.transitionStart = this.stateStart;

      this.root = document.createElement("span");
      this.root.className = "illustrated-avatar-layer";
      this.root.setAttribute("aria-hidden", "true");
      Object.assign(this.root.style, {
        position: "absolute",
        display: "block",
        overflow: "hidden",
        pointerEvents: "none",
        zIndex: "2",
      });

      this.previous = this.makeFrame("illustrated-avatar-frame illustrated-avatar-frame--previous");
      this.current = this.makeFrame("illustrated-avatar-frame illustrated-avatar-frame--current");
      this.root.appendChild(this.previous);
      this.root.appendChild(this.current);

      const parent = canvas.parentElement;
      if (parent && window.getComputedStyle(parent).position === "static") {
        parent.style.position = "relative";
      }
      if (parent) parent.appendChild(this.root);
      this.syncGeometry();
    }

    makeFrame(className) {
      const frame = document.createElement("span");
      frame.className = className;
      Object.assign(frame.style, {
        position: "absolute",
        inset: "0",
        display: "block",
        pointerEvents: "none",
        backgroundRepeat: "no-repeat",
        backgroundOrigin: "border-box",
        backgroundClip: "border-box",
        transformOrigin: "50% 72%",
        willChange: "transform, opacity",
      });
      return frame;
    }

    syncGeometry() {
      const width = this.canvas.offsetWidth || this.canvas.width;
      const height = this.canvas.offsetHeight || this.canvas.height;
      this.width = width;
      this.height = height;
      this.root.style.left = this.canvas.offsetLeft + "px";
      this.root.style.top = this.canvas.offsetTop + "px";
      this.root.style.width = width + "px";
      this.root.style.height = height + "px";
      [this.previous, this.current].forEach(function (frame) {
        frame.style.backgroundSize = (width * 2) + "px " + (height * 9) + "px";
      });
    }

    enable(spriteUrl, state, now) {
      const image = 'url("' + spriteUrl + '")';
      this.previous.style.backgroundImage = image;
      this.current.style.backgroundImage = image;
      this.canvas.style.opacity = "0";
      this.enabled = true;
      this.state = state;
      this.previousState = state;
      this.stateStart = now;
      this.transitionStart = now - TRANSITION_MS;
      this.syncGeometry();
    }

    setState(next, now) {
      if (next === this.state) return;
      this.previousState = this.state;
      this.state = next;
      this.transitionStart = now;
      this.stateStart = now;
    }

    setSpriteFrame(element, state, lineFrame) {
      const row = Object.prototype.hasOwnProperty.call(ROW, state) ? ROW[state] : ROW.error;
      element.style.backgroundPosition =
        (-lineFrame * this.width) + "px " + (-row * this.height) + "px";
    }

    draw(now, lineFrame) {
      if (!this.enabled) return;
      this.setSpriteFrame(this.previous, this.previousState, lineFrame);
      this.setSpriteFrame(this.current, this.state, lineFrame);

      if (reducedMotion) {
        this.previous.style.opacity = "0";
        this.current.style.opacity = this.state === "offline" ? "0.66" : "1";
        this.current.style.transform = "none";
        return;
      }

      const rawProgress = (now - this.transitionStart) / TRANSITION_MS;
      if (rawProgress < 1) {
        const p = easeOutCubic(rawProgress);
        this.previous.style.opacity = String(1 - p);
        this.previous.style.transform =
          "translateY(" + (-1.2 * p).toFixed(2) + "px) scale(" + (1 - 0.012 * p).toFixed(4) + ")";
        this.current.style.opacity = String(p);
        this.current.style.transform =
          "translateY(" + (2.2 * (1 - p)).toFixed(2) + "px) scale(" + (0.985 + 0.015 * p).toFixed(4) + ")";
        return;
      }

      this.previous.style.opacity = "0";
      const motion = stateMotion(this.state, now - this.stateStart);
      this.current.style.opacity = String(motion.opacity);
      this.current.style.transform = motionTransform(motion);
    }
  }

  const layers = canvases.map(function (canvas) { return new IllustratedAvatarLayer(canvas); });

  function currentState() {
    const state = document.documentElement.dataset.state || "restarting";
    return Object.prototype.hasOwnProperty.call(ROW, state) ? state : "error";
  }

  const observer = new MutationObserver(function (records) {
    if (!records.some(function (record) { return record.attributeName === "data-state"; })) return;
    const state = currentState();
    const now = performance.now();
    layers.forEach(function (layer) { layer.setState(state, now); });
  });
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-state"] });

  window.addEventListener("resize", function () {
    layers.forEach(function (layer) { layer.syncGeometry(); });
  });

  const sprite = new Image();
  sprite.onload = function () {
    const state = currentState();
    const now = performance.now();
    layers.forEach(function (layer) { layer.enable(sprite.src, state, now); });
    requestAnimationFrame(animate);
  };
  sprite.onerror = function () {
    // Deliberately do nothing: app.js' existing canvas avatar remains visible.
    console.warn("LocalPilot: illustrated avatar asset unavailable; using pixel fallback.");
  };
  sprite.src = "avatar/sprite.png";

  function animate(now) {
    // Line boil is deliberately independent of state motion.  Offline and
    // reduced-motion modes freeze it; otherwise it quietly alternates redraws.
    const state = currentState();
    const lineFrame = (reducedMotion || state === "offline")
      ? 0
      : (Math.floor(now / LINE_BOIL_MS) % 2);
    layers.forEach(function (layer) { layer.draw(now, lineFrame); });
    requestAnimationFrame(animate);
  }
})();
