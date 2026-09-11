"use strict";

/* LocalPilot illustrated avatar: real frame-sequence animation.
 *
 * app.js still owns runtime state and the original pixel canvas remains the
 * fail-safe.  This layer loads a verified manifest of per-state PNG strips.
 * Each strip contains enter frames, a state-specific loop, and exit frames.
 *
 * No whole-character CSS nudging is used to fake activity.  Thinking really
 * writes, working really types, speaking changes mouth/gesture frames, success
 * pumps a fist, sleeping breathes, etc.  State changes play the old exit and
 * new enter sequences simultaneously during a short crossfade.
 */
(function () {
  "use strict";

  const MANIFEST_URL = "avatar/anim/animation-manifest.json";
  const ASSET_ROOT = "avatar/anim/";
  const REQUIRED_STATES = [
    "idle", "listening", "thinking", "researching", "working", "speaking",
    "success", "uncertain", "error", "learning", "restarting", "sleeping", "offline",
  ];
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const canvases = [
    document.getElementById("avatar-dock"),
    document.getElementById("avatar-header"),
  ].filter(Boolean);

  if (!canvases.length) return;

  function clamp01(value) {
    return Math.max(0, Math.min(1, value));
  }

  function normalizeState(state) {
    return REQUIRED_STATES.includes(state) ? state : "error";
  }

  function frameForElapsed(spec, elapsedMs) {
    if (reducedMotion) return Number(spec.representative_frame || spec.loop_start || 0);
    const start = Number(spec.loop_start);
    const end = Number(spec.loop_end);
    const count = Math.max(1, end - start + 1);
    const step = Math.floor(Math.max(0, elapsedMs) / Number(spec.frame_ms));
    return start + (step % count);
  }

  function transitionFrame(spec, startKey, endKey, progress) {
    const start = Number(spec[startKey]);
    const end = Number(spec[endKey]);
    const count = Math.max(1, end - start + 1);
    const index = Math.min(count - 1, Math.floor(clamp01(progress) * count));
    return start + index;
  }

  class IllustratedAvatarLayer {
    constructor(canvas, manifest) {
      this.canvas = canvas;
      this.manifest = manifest;
      this.frameSize = Number(manifest.frame_size);
      this.transitionMs = Number(manifest.transition_ms);
      this.enabled = false;

      this.state = "restarting";
      this.previousState = "restarting";
      this.stateStart = performance.now();
      this.transitionStart = this.stateStart - this.transitionMs;

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
        transform: "none",
        willChange: "background-position, opacity",
      });
      return frame;
    }

    spec(state) {
      return this.manifest.states[normalizeState(state)];
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
    }

    setFrame(element, state, frameIndex) {
      const spec = this.spec(state);
      const frames = Number(spec.frames);
      element.style.backgroundImage = 'url("' + ASSET_ROOT + spec.file + '")';
      element.style.backgroundSize = (this.width * frames) + "px " + this.height + "px";
      element.style.backgroundPosition = (-frameIndex * this.width) + "px 0px";
    }

    enable(state, now) {
      this.canvas.style.opacity = "0";
      this.enabled = true;
      this.state = normalizeState(state);
      this.previousState = this.state;
      this.stateStart = now;
      this.transitionStart = now - this.transitionMs;
      this.syncGeometry();
      const spec = this.spec(this.state);
      this.setFrame(this.current, this.state, Number(spec.representative_frame));
      this.current.style.opacity = "1";
      this.previous.style.opacity = "0";
    }

    setState(next, now) {
      next = normalizeState(next);
      if (next === this.state) return;
      this.previousState = this.state;
      this.state = next;
      this.transitionStart = now;
      this.stateStart = now + this.transitionMs;
    }

    draw(now) {
      if (!this.enabled) return;

      if (reducedMotion) {
        const spec = this.spec(this.state);
        this.setFrame(this.current, this.state, Number(spec.representative_frame));
        this.current.style.opacity = this.state === "offline" ? "0.72" : "1";
        this.previous.style.opacity = "0";
        return;
      }

      const transitionProgress = (now - this.transitionStart) / this.transitionMs;
      if (transitionProgress < 1) {
        const p = clamp01(transitionProgress);
        const oldSpec = this.spec(this.previousState);
        const newSpec = this.spec(this.state);
        const oldFrame = transitionFrame(oldSpec, "exit_start", "exit_end", p);
        const newFrame = transitionFrame(newSpec, "enter_start", "enter_end", p);

        this.setFrame(this.previous, this.previousState, oldFrame);
        this.setFrame(this.current, this.state, newFrame);
        this.previous.style.opacity = String(1 - p);
        this.current.style.opacity = String(p);
        return;
      }

      this.previous.style.opacity = "0";
      const spec = this.spec(this.state);
      const frame = frameForElapsed(spec, now - this.stateStart);
      this.setFrame(this.current, this.state, frame);
      this.current.style.opacity = "1";
    }
  }

  function currentState() {
    return normalizeState(document.documentElement.dataset.state || "restarting");
  }

  async function loadManifestAndAssets() {
    const response = await fetch(MANIFEST_URL, { cache: "no-store" });
    if (!response.ok) throw new Error("animation manifest unavailable");
    const manifest = await response.json();

    if (
      Number(manifest.version) !== 2 ||
      Number(manifest.frame_size) !== 128 ||
      !manifest.states
    ) {
      throw new Error("invalid animation manifest");
    }

    for (const state of REQUIRED_STATES) {
      const spec = manifest.states[state];
      if (!spec || !spec.file || Number(spec.frames) < 8) {
        throw new Error("missing animation state: " + state);
      }
    }

    await Promise.all(REQUIRED_STATES.map(function (state) {
      return new Promise(function (resolve, reject) {
        const spec = manifest.states[state];
        const image = new Image();
        image.onload = function () {
          if (
            image.naturalWidth !== Number(manifest.frame_size) * Number(spec.frames) ||
            image.naturalHeight !== Number(manifest.frame_size)
          ) {
            reject(new Error("invalid animation strip dimensions: " + state));
            return;
          }
          resolve();
        };
        image.onerror = reject;
        image.src = ASSET_ROOT + spec.file;
      });
    }));

    return manifest;
  }

  loadManifestAndAssets().then(function (manifest) {
    const layers = canvases.map(function (canvas) {
      return new IllustratedAvatarLayer(canvas, manifest);
    });

    const observer = new MutationObserver(function (records) {
      if (!records.some(function (record) { return record.attributeName === "data-state"; })) return;
      const state = currentState();
      const now = performance.now();
      layers.forEach(function (layer) { layer.setState(state, now); });
    });
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-state"],
    });

    window.addEventListener("resize", function () {
      layers.forEach(function (layer) { layer.syncGeometry(); });
    });

    const state = currentState();
    const now = performance.now();
    layers.forEach(function (layer) { layer.enable(state, now); });

    function animate(timestamp) {
      layers.forEach(function (layer) { layer.draw(timestamp); });
      requestAnimationFrame(animate);
    }
    requestAnimationFrame(animate);
  }).catch(function (error) {
    // Deliberately leave app.js' known-good pixel avatar visible.
    console.warn("LocalPilot: illustrated animation unavailable; using pixel fallback.", error);
  });
})();
