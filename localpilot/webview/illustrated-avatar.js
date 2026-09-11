"use strict";

/* LocalPilot illustrated avatar: real frame-sequence animation.
 *
 * app.js still owns runtime state and the original pixel canvas remains the
 * fail-safe. This layer loads one compact animation atlas plus a manifest.
 *
 * Each state has enter frames, a genuine state-specific animated loop, and
 * exit frames. State changes play the old exit and new enter sequences at the
 * same time during a short crossfade. No whole-character CSS nudging is used
 * to fake writing, typing, talking, breathing, celebrating, or sleeping.
 */
(function () {
  "use strict";

  const MANIFEST_URL = "avatar/anim/animation-manifest.json";
  const ATLAS_URL = "avatar/anim/avatar-animation.png";
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
        backgroundImage: 'url("' + ATLAS_URL + '")',
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
      const atlasWidth = width * Number(this.manifest.columns);
      const atlasHeight = height * Number(this.manifest.rows);
      [this.previous, this.current].forEach(function (frame) {
        frame.style.backgroundSize = atlasWidth + "px " + atlasHeight + "px";
      });
    }

    setFrame(element, state, frameIndex) {
      const spec = this.spec(state);
      element.style.backgroundPosition =
        (-frameIndex * this.width) + "px " + (-Number(spec.row) * this.height) + "px";
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
        this.setFrame(
          this.previous,
          this.previousState,
          transitionFrame(oldSpec, "exit_start", "exit_end", p)
        );
        this.setFrame(
          this.current,
          this.state,
          transitionFrame(newSpec, "enter_start", "enter_end", p)
        );
        this.previous.style.opacity = String(1 - p);
        this.current.style.opacity = String(p);
        return;
      }

      this.previous.style.opacity = "0";
      const spec = this.spec(this.state);
      this.setFrame(this.current, this.state, frameForElapsed(spec, now - this.stateStart));
      this.current.style.opacity = "1";
    }
  }

  function currentState() {
    return normalizeState(document.documentElement.dataset.state || "restarting");
  }

  async function loadManifestAndAtlas() {
    const response = await fetch(MANIFEST_URL, { cache: "no-store" });
    if (!response.ok) throw new Error("animation manifest unavailable");
    const manifest = await response.json();

    if (
      Number(manifest.version) !== 3 ||
      Number(manifest.frame_size) !== 128 ||
      Number(manifest.columns) < 1 ||
      Number(manifest.rows) < REQUIRED_STATES.length ||
      !manifest.states ||
      !manifest.atlas ||
      manifest.atlas.file !== "avatar-animation.png"
    ) {
      throw new Error("invalid animation manifest");
    }

    for (const state of REQUIRED_STATES) {
      const spec = manifest.states[state];
      if (!spec || Number(spec.frames) < 8 || Number(spec.row) < 0) {
        throw new Error("missing animation state: " + state);
      }
    }

    await new Promise(function (resolve, reject) {
      const image = new Image();
      image.onload = function () {
        if (
          image.naturalWidth !== Number(manifest.frame_size) * Number(manifest.columns) ||
          image.naturalHeight !== Number(manifest.frame_size) * Number(manifest.rows)
        ) {
          reject(new Error("invalid animation atlas dimensions"));
          return;
        }
        resolve();
      };
      image.onerror = reject;
      image.src = ATLAS_URL;
    });

    return manifest;
  }

  loadManifestAndAtlas().then(function (manifest) {
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
