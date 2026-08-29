"use strict";
/* LocalPilot desktop companion — production frontend.
 *
 * Talks directly to the real broker over HTTP (sessions/messages/events),
 * per the design handoff: this file owns all conversation data. Window
 * chrome (resize, always-on-top, startup registration) goes through the
 * small js_api bridge exposed by webview_app.py instead — see the
 * `bridge()` helper below. The two are kept deliberately separate.
 *
 * Bootstraps via window.__initLocalPilot(payload), called once by Python
 * (webview_app.py) after the window's 'loaded' event, with
 * { baseUrl, token, hasConfigPath }.
 */


(function () {
  "use strict";

  /* ======================================================================
     §1/§3 — state → color + motion character (mirrors design.md exactly)
     ====================================================================== */
  const STATES = ["idle","listening","thinking","working","speaking","success","uncertain","error","restarting","sleeping","offline"];

  const COLOR = {
    idle:        { core: [99,217,184] },
    listening:   { core: [126,234,209] },
    thinking:    { core: [240,194,78] },
    working:     { core: [95,168,255] },
    speaking:    { core: [185,140,255] },
    success:     { core: [111,222,142] },
    uncertain:   { core: [240,162,78] },
    error:       { core: [255,107,107] },
    restarting:  { core: [133,146,168] },
    sleeping:    { core: [99,217,184] },
    offline:     { core: [75,85,104] },
  };
  // breathing period in seconds; 0 = no continuous breathing (state has its own special motion)
  const BREATHE_PERIOD = {
    idle: 4, listening: 1.6, thinking: 4, working: 4, speaking: 4,
    success: 0, uncertain: 4.6, error: 0, restarting: 0, sleeping: 8, offline: 0,
  };
  const STATE_LABEL = {
    idle: "Idle", listening: "Listening", thinking: "Thinking", working: "Working",
    speaking: "Speaking", success: "Success", uncertain: "Uncertain", error: "Error",
    restarting: "Restarting", sleeping: "Sleeping", offline: "Offline",
  };
  const LIVE_PHRASE = {
    idle: "LocalPilot is idle.", listening: "LocalPilot is listening.",
    thinking: "LocalPilot is thinking.", working: "LocalPilot is working.",
    speaking: "LocalPilot is speaking.", success: "LocalPilot finished successfully.",
    uncertain: "LocalPilot is unsure about its answer.", error: "LocalPilot ran into an error.",
    restarting: "LocalPilot's runtime is restarting.", sleeping: "LocalPilot is sleeping.",
    offline: "LocalPilot's broker is unreachable.",
  };

  function lerp(a, b, t) { return a + (b - a) * t; }
  function lerpColor(c1, c2, t) {
    return [Math.round(lerp(c1[0],c2[0],t)), Math.round(lerp(c1[1],c2[1],t)), Math.round(lerp(c1[2],c2[2],t))];
  }
  function rgbStr(c, a) { return "rgba(" + c[0] + "," + c[1] + "," + c[2] + "," + (a===undefined?1:a) + ")"; }

  /* ======================================================================
     §3/§7 — Avatar: a small canvas-rendered pixel creature.
     One class instance per <canvas>; all instances share the same global
     `state` via setState() calls broadcast from the state store below, so
     the dock avatar and the header avatar are always in sync (§9 — same
     underlying state driving two rendered instances of one component).
     ====================================================================== */
  class Avatar {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.cols = 15; this.rows = 15;
      this.cell = canvas.width / this.cols;
      this.state = "restarting";
      this.prevState = "restarting";
      this.transitionStart = performance.now();
      this.oneShotStart = null;
      this.startTime = performance.now();
      this.flickerSeed = Math.random() * 1000;
      this.blinkOffset = Math.random() * 400; // asymmetry for "uncertain"
      this.ventLevel = 0; // driven externally during "speaking" to follow text reveal
      this.reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    }

    setState(next) {
      if (next === this.state) return;
      this.prevState = this.state;
      this.state = next;
      this.transitionStart = performance.now();
      if (next === "success" || next === "error") this.oneShotStart = performance.now();
    }

    currentColor(now) {
      const t = this.reduced ? 1 : Math.min(1, (now - this.transitionStart) / 400);
      return lerpColor(COLOR[this.prevState].core, COLOR[this.state].core, t);
    }

    draw(now) {
      const ctx = this.ctx, cols = this.cols, rows = this.rows, cell = this.cell;
      ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      const elapsed = (now - this.startTime) / 1000;
      const state = this.state;
      const color = this.currentColor(now);

      // ---- breathing (transform: scale-equivalent via radius modulation) ----
      let breathe = 1;
      const period = BREATHE_PERIOD[state];
      if (!this.reduced && period > 0) {
        breathe = 1 + 0.035 * Math.sin((elapsed / period) * Math.PI * 2);
      }

      // ---- one-shot motion: error shake / success bloom ----
      let shakeX = 0, bloom = 0;
      if (state === "error" && this.oneShotStart !== null) {
        const t = now - this.oneShotStart;
        if (t < 240) shakeX = Math.sin(t / 18) * 5 * (1 - t / 240);
      }
      if (state === "success" && this.oneShotStart !== null) {
        const t = (now - this.oneShotStart) / 900;
        if (t < 1) bloom = Math.sin(Math.min(t,1) * Math.PI);
      }

      const cx = cols / 2, cy = rows / 2 + 0.4;
      const rx = 5.1 * breathe, ryTop = 4.5 * breathe, ryBot = 6.0 * breathe;
      const offline = state === "offline";
      const dim = state === "sleeping" ? 0.32 : (state === "offline" ? 0.5 : 1);

      ctx.save();
      ctx.translate(shakeX, 0);
      if (!this.reduced) ctx.globalAlpha = dim;

      // glow behind the whole body (canvas-native soft shadow — no extra layers needed)
      if (!offline) {
        ctx.shadowColor = rgbStr(color, 0.9);
        ctx.shadowBlur = cell * (1.4 + bloom * 1.6);
      } else {
        ctx.shadowBlur = 0;
      }

      // ---- body: superellipse-ish blob, wider/flatter at the bottom (grounded, sitting) ----
      ctx.fillStyle = rgbStr(color, 1);
      for (let gy = 0; gy < rows; gy++) {
        for (let gx = 0; gx < cols; gx++) {
          const u = (gx + 0.5 - cx) / rx;
          const vRadius = gy < cy ? ryTop : ryBot;
          const v = (gy + 0.5 - cy) / vRadius;
          if (u * u + v * v <= 1 + bloom * 0.06) {
            ctx.fillRect(gx * cell, gy * cell, cell - 0.6, cell - 0.6);
          }
        }
      }
      ctx.shadowBlur = 0;

      // ---- eyes: the primary expressive device (non-human — no literal mouth) ----
      if (!offline) {
        const eyeRow = Math.round(cy) - (state === "working" ? 1 : 2);
        const eyeOffset = 2;
        const accent = state === "sleeping" ? rgbStr(color, 0.7) : "rgba(240,251,255,0.95)";
        ctx.fillStyle = accent;

        const drawEye = (colCenter, closed, wide) => {
          const x = colCenter * cell;
          if (closed) {
            ctx.fillRect(x - cell * 0.9, eyeRow * cell + cell * 0.4, cell * 1.8, cell * 0.35);
          } else {
            const w = wide ? 1.7 : 1;
            ctx.fillRect(x - (cell * w) / 2, eyeRow * cell, cell * w, cell * (state==="error"?0.4:1));
          }
        };

        if (state === "restarting") {
          const on = Math.floor(elapsed) % 2 === 0; // stepped, unequal — deliberately mechanical
          if (on) { drawEye(cx - eyeOffset, false); drawEye(cx + eyeOffset, false); }
        } else if (state === "sleeping" || state === "offline") {
          drawEye(cx - eyeOffset, true); drawEye(cx + eyeOffset, true);
        } else if (state === "error") {
          drawEye(cx - eyeOffset, false); drawEye(cx + eyeOffset, false);
        } else if (state === "uncertain") {
          const lClosed = ((elapsed*1000 + this.blinkOffset) % 2600) < 180;
          const rClosed = ((elapsed*1000) % 2600) < 180;
          drawEye(cx - eyeOffset, lClosed); drawEye(cx + eyeOffset, rClosed);
        } else if (state === "thinking") {
          const flick = Math.sin(elapsed * 3.1 + this.flickerSeed) > 0.4;
          drawEye(cx - eyeOffset, false, false);
          drawEye(cx + eyeOffset + (flick ? 0.4 : 0), false, false);
        } else if (state === "success" && this.oneShotStart !== null && (now - this.oneShotStart) < 260) {
          drawEye(cx - eyeOffset, true); drawEye(cx + eyeOffset, true);
        } else if (state === "listening") {
          drawEye(cx - eyeOffset, false, true); drawEye(cx + eyeOffset, false, true);
        } else {
          drawEye(cx - eyeOffset, false); drawEye(cx + eyeOffset, false);
        }
      }

      // ---- vent: lights up only while speaking, brightness follows ventLevel ----
      if (state === "speaking") {
        const level = 0.35 + 0.65 * this.ventLevel;
        ctx.fillStyle = rgbStr(color, level);
        ctx.shadowColor = rgbStr(color, 0.8);
        ctx.shadowBlur = cell * 1.2 * level;
        const ventRow = Math.round(cy) + 3;
        ctx.fillRect((cx - 1.3) * cell, ventRow * cell, 2.6 * cell, cell * 0.5);
        ctx.shadowBlur = 0;
      }

      ctx.restore();

      // ---- motes: ambient orbit while working (not limbs — stays non-anthropomorphic) ----
      if (state === "working") {
        for (let i = 0; i < 3; i++) {
          const angle = (elapsed / 6) * Math.PI * 2 + (i * (Math.PI * 2 / 3));
          const orbitR = (rx + 2.1) * cell;
          const mx = cx * cell + Math.cos(angle) * orbitR;
          const my = cy * cell + Math.sin(angle) * orbitR * 0.85;
          ctx.save();
          ctx.shadowColor = rgbStr(color, 0.9);
          ctx.shadowBlur = cell * 1.1;
          ctx.fillStyle = rgbStr(color, 0.85);
          ctx.beginPath(); ctx.arc(mx, my, cell * 0.32, 0, Math.PI * 2); ctx.fill();
          ctx.restore();
        }
      }
    }
  }
    /* class Avatar ends here — real app wiring continues below */

  /* ======================================================================
     Broker client — direct HTTP to the real broker, per the design doc.
     Set once by window.__initLocalPilot(); nothing runs before that.
     ====================================================================== */
  let BASE_URL = null;
  let TOKEN = null;
  let hasConfigPath = false;

  async function api(method, path, body) {
    const headers = { Accept: "application/json" };
    if (TOKEN) headers["Authorization"] = "Bearer " + TOKEN;
    let payload;
    if (body !== undefined) {
      headers["Content-Type"] = "application/json; charset=utf-8";
      payload = JSON.stringify(body);
    }
    const response = await fetch(BASE_URL + path, { method: method, headers: headers, body: payload });
    if (!response.ok) {
      const err = new Error("HTTP " + response.status + " for " + method + " " + path);
      err.status = response.status;
      throw err;
    }
    const text = await response.text();
    return text ? JSON.parse(text) : {};
  }

  async function checkHealth() {
    try {
      const res = await fetch(BASE_URL + "/health", { method: "GET" });
      if (!res.ok) return { reachable: true, runtime: "error" };
      const data = await res.json();
      return { reachable: true, runtime: data.runtime || "running" };
    } catch (e) {
      return { reachable: false, runtime: null };
    }
  }

  function sleep(ms) { return new Promise(function (resolve) { setTimeout(resolve, ms); }); }

  /* ======================================================================
     Window-chrome bridge (js_api, webview_app.py) — window state only.
     Conversation data never goes through this; see file header.
     ====================================================================== */
  async function bridge(method) {
    const args = Array.prototype.slice.call(arguments, 1);
    try {
      if (!window.pywebview || !window.pywebview.api || typeof window.pywebview.api[method] !== "function") {
        console.warn("LocalPilot: bridge method not available yet:", method);
        return null;
      }
      return await window.pywebview.api[method].apply(null, args);
    } catch (e) {
      console.warn("LocalPilot: bridge call failed:", method, e);
      return null;
    }
  }

  /* ======================================================================
     DOM references
     ====================================================================== */
  const app = document.getElementById("app");
  const panelEl = document.getElementById("panel");
  const dockEl = document.getElementById("dock");
  const previewLine = document.getElementById("preview-line");
  const stateLabel = document.getElementById("state-label");
  const liveRegion = document.getElementById("live-region");
  const messageStream = document.getElementById("message-stream");
  const activityStrip = document.getElementById("activity-strip");
  const activityText = document.getElementById("activity-text");
  const activitySteps = document.getElementById("activity-steps");
  const composerWrap = document.getElementById("composer");
  const composerInput = document.getElementById("composer-input");
  const sendBtn = document.getElementById("send-btn");
  const historyToggle = document.getElementById("history-toggle");
  const historySheet = document.getElementById("history-sheet");
  const historyClose = document.getElementById("history-close");
  const historyNew = document.getElementById("history-new");
  const historyList = document.getElementById("history-list");
  const settingsToggle = document.getElementById("settings-toggle");
  const settingsPopover = document.getElementById("settings-popover");
  const collapseBtn = document.getElementById("collapse-btn");
  const restartBtn = document.getElementById("restart-btn");
  const startupToggle = document.getElementById("toggle-startup");
  const ontopToggle = document.getElementById("toggle-ontop");
  const openConfigBtn = document.getElementById("open-config");
  const settingsStatusText = document.getElementById("settings-status-text");
  const settingsDot = document.getElementById("settings-dot");
  const systemToggle = document.getElementById("system-toggle");
  const systemHealthDot = document.getElementById("system-health-dot");
  const systemPanel = document.getElementById("system-panel");
  const systemClose = document.getElementById("system-close");
  const systemLoading = document.getElementById("system-loading");
  const systemContent = document.getElementById("system-content");
  const systemHeroDot = document.getElementById("system-hero-dot");
  const systemHealthLabel = document.getElementById("system-health-label");
  const systemHealthSummary = document.getElementById("system-health-summary");
  const systemUpdated = document.getElementById("system-updated");
  const systemCpuValue = document.getElementById("system-cpu-value");
  const systemCpuDetail = document.getElementById("system-cpu-detail");
  const systemGpuValue = document.getElementById("system-gpu-value");
  const systemGpuDetail = document.getElementById("system-gpu-detail");
  const systemMemoryValue = document.getElementById("system-memory-value");
  const systemMemoryDetail = document.getElementById("system-memory-detail");
  const systemTemperatureValue = document.getElementById("system-temperature-value");
  const systemTemperatureDetail = document.getElementById("system-temperature-detail");
  const systemStorageValue = document.getElementById("system-storage-value");
  const systemStorageDetail = document.getElementById("system-storage-detail");
  const systemVramValue = document.getElementById("system-vram-value");
  const systemVramDetail = document.getElementById("system-vram-detail");
  const systemInferenceValue = document.getElementById("system-inference-value");
  const systemInferenceDetail = document.getElementById("system-inference-detail");
  const systemSignalList = document.getElementById("system-signal-list");
  const systemProcessList = document.getElementById("system-process-list");
  const systemRefresh = document.getElementById("system-refresh");

  const dockAvatar = new Avatar(document.getElementById("avatar-dock"));
  const headerAvatar = new Avatar(document.getElementById("avatar-header"));
  const avatars = [dockAvatar, headerAvatar];

  /* ======================================================================
     State store — identical contract to the verified prototype (§10 of the
     design doc): drives both avatar instances, the text label, and the
     accent CSS variables from one place.
     ====================================================================== */
  let currentState = null;

  function setGlobalState(next) {
    if (!STATE_LABEL[next] || next === currentState) return;
    currentState = next;
    avatars.forEach(function (a) { a.setState(next); });
    stateLabel.textContent = STATE_LABEL[next];
    document.documentElement.dataset.state = next;
    liveRegion.textContent = LIVE_PHRASE[next];
    if (next === "idle") armSleepTimer();
  }

  function rafLoop(now) {
    avatars.forEach(function (a) { a.draw(now); });
    requestAnimationFrame(rafLoop);
  }
  requestAnimationFrame(rafLoop);

  /* ======================================================================
     Sleeping (client-local, §10.2): idle + no window focus for a while.
     ====================================================================== */
  let sleepTimer = null;
  const SLEEP_AFTER_MS = 10 * 60 * 1000;
  function armSleepTimer() {
    clearTimeout(sleepTimer);
    sleepTimer = setTimeout(function () {
      if (currentState === "idle" && !document.hasFocus()) setGlobalState("sleeping");
    }, SLEEP_AFTER_MS);
  }
  window.addEventListener("focus", function () {
    if (currentState === "sleeping") setGlobalState("idle");
    armSleepTimer();
  });
  window.addEventListener("blur", armSleepTimer);

  /* ======================================================================
     Compact <-> expanded (§2.1/§2.2). The CSS crossfade is instant/local;
     the actual native window resize goes through the bridge. Note: unlike
     the browser prototype, pywebview does not animate window resizes, so
     the resize itself will snap rather than glide — see delivery notes.
     ====================================================================== */
  function expand() {
    app.classList.add("is-expanded");
    bridge("expand");
    composerInput.focus();
  }
  function collapse() {
    app.classList.remove("is-expanded");
    bridge("collapse");
    closeSettings();
    closeHistory();
    closeSystemPanel();
  }
  dockEl.addEventListener("click", expand);
  dockEl.addEventListener("keydown", function (e) {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); expand(); }
  });
  collapseBtn.addEventListener("click", collapse);

  /* ======================================================================
     History sheet (§2.3) — backed by the real /v1/sessions list.
     ====================================================================== */
  function openHistory() { closeSystemPanel(); historySheet.classList.add("is-open"); historySheet.setAttribute("aria-hidden", "false"); loadSessionList(); }
  function closeHistory() { historySheet.classList.remove("is-open"); historySheet.setAttribute("aria-hidden", "true"); }
  historyToggle.addEventListener("click", function () {
    if (historySheet.classList.contains("is-open")) closeHistory();
    else { closeSettings(); openHistory(); }
  });
  historyClose.addEventListener("click", closeHistory);
  historyNew.addEventListener("click", async function () {
    try {
      const created = await api("POST", "/v1/sessions", {});
      await switchSession(created.session.id);
      closeHistory();
      composerInput.focus();
    } catch (e) {
      console.warn("LocalPilot: failed to create conversation", e);
    }
  });

  function relativeTime(isoString) {
    const then = new Date(isoString).getTime();
    if (isNaN(then)) return "";
    const diffMin = Math.round((Date.now() - then) / 60000);
    if (diffMin < 1) return "Just now";
    if (diffMin < 60) return diffMin + " min ago";
    const diffHr = Math.round(diffMin / 60);
    if (diffHr < 24) return diffHr + " hr ago";
    const diffDay = Math.round(diffHr / 24);
    if (diffDay === 1) return "Yesterday";
    if (diffDay < 7) return diffDay + " days ago";
    return new Date(isoString).toLocaleDateString();
  }

  function renderHistoryList(sessions) {
    historyList.replaceChildren();
    sessions.forEach(function (s) {
      const row = document.createElement("div");
      row.className = "session-row" + (s.id === activeSessionId ? " is-active" : "");
      const dot = document.createElement("span"); dot.className = "dot";
      const meta = document.createElement("div");
      const title = document.createElement("div"); title.className = "session-row__title";
      title.textContent = s.title || "Untitled conversation";
      const time = document.createElement("div"); time.className = "session-row__time";
      time.textContent = relativeTime(s.updated_at);
      meta.appendChild(title); meta.appendChild(time);
      row.appendChild(dot); row.appendChild(meta);
      row.tabIndex = 0;
      row.addEventListener("click", function () { switchSession(s.id); closeHistory(); });
      row.addEventListener("keydown", function (e) {
        if (e.key === "Enter") { switchSession(s.id); closeHistory(); }
      });
      historyList.appendChild(row);
    });
  }

  async function loadSessionList() {
    try {
      const data = await api("GET", "/v1/sessions");
      const sessions = (data.sessions || []).slice().sort(function (a, b) {
        return new Date(b.updated_at) - new Date(a.updated_at);
      });
      renderHistoryList(sessions);
      return sessions;
    } catch (e) {
      console.warn("LocalPilot: failed to load session list", e);
      return [];
    }
  }

  /* ======================================================================
     Settings popover (§2.4) — status from /health, restart via the real
     endpoint, window-chrome toggles via the bridge, config via the bridge.
     ====================================================================== */
  function openSettings() { closeSystemPanel(); settingsPopover.classList.add("is-open"); }
  function closeSettings() { settingsPopover.classList.remove("is-open"); }
  settingsToggle.addEventListener("click", function () {
    if (settingsPopover.classList.contains("is-open")) closeSettings();
    else { closeHistory(); openSettings(); }
  });
  document.addEventListener("click", function (e) {
    const clickedToggle = e.target === settingsToggle || settingsToggle.contains(e.target);
    if (!settingsPopover.contains(e.target) && !clickedToggle) closeSettings();
  });

  startupToggle.addEventListener("click", async function () {
    const wantOn = !startupToggle.classList.contains("is-on");
    const result = await bridge("set_start_with_windows", wantOn);
    if (result && result.ok) {
      startupToggle.classList.toggle("is-on", wantOn);
      startupToggle.setAttribute("aria-checked", String(wantOn));
    }
  });
  ontopToggle.addEventListener("click", async function () {
    const wantOn = !ontopToggle.classList.contains("is-on");
    const result = await bridge("set_always_on_top", wantOn);
    if (result && result.ok) {
      ontopToggle.classList.toggle("is-on", wantOn);
      ontopToggle.setAttribute("aria-checked", String(wantOn));
    }
  });
  openConfigBtn.addEventListener("click", function (e) {
    bridge("open_config_file");
    const btn = e.currentTarget;
    btn.classList.add("is-activated");
    setTimeout(function () { btn.classList.remove("is-activated"); }, 220);
  });
  restartBtn.addEventListener("click", async function () {
    settingsStatusText.textContent = "Restarting\u2026";
    try {
      await api("POST", "/v1/runtime/restart");
      // the real transition (restarting -> ready -> idle) arrives via the
      // event stream; this call just asks for it to happen.
    } catch (e) {
      console.warn("LocalPilot: restart request failed", e);
      settingsStatusText.textContent = "Restart request failed";
    }
  });

  /* ======================================================================
     SystemSense glance panel — authenticated, bounded summary only. The
     runtime owns collection; this UI never starts a collector or exposes a
     control surface. Poll quickly only while the panel is visible.
     ====================================================================== */
  const SYSTEM_HEALTH_CLASSES = ["is-good", "is-degraded", "is-critical", "is-unknown"];
  const SYSTEM_HEALTH_LABELS = {
    good: "Healthy",
    degraded: "Needs attention",
    critical: "Critical",
    unknown: "Waiting for data",
  };
  const SYSTEM_METRIC_LABELS = {
    "cpu.percent": "CPU load",
    "cpu.frequency_mhz": "CPU frequency",
    "memory.percent": "memory use",
    "memory.available_gb": "available memory",
    "storage.read_mb_s": "storage reads",
    "storage.write_mb_s": "storage writes",
    "network.send_mbps": "network upload",
    "network.receive_mbps": "network download",
    "gpu.utilization_percent": "GPU load",
    "thermal.max_c": "temperature",
    "vram.used_mb": "VRAM use",
  };
  let systemPanelOpen = false;
  let systemRefreshTimer = null;
  let systemRefreshInFlight = false;
  let lastSystemSummary = null;

  function finiteNumber(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function setSystemHealthClass(element, health) {
    SYSTEM_HEALTH_CLASSES.forEach(function (name) { element.classList.remove(name); });
    element.classList.add("is-" + (SYSTEM_HEALTH_LABELS[health] ? health : "unknown"));
  }

  function setSystemMetric(valueElement, detailElement, value, suffix, detail, digits) {
    const number = finiteNumber(value);
    if (number === null) {
      valueElement.textContent = "—";
      detailElement.textContent = "Not reported";
      return;
    }
    valueElement.textContent = number.toFixed(digits || 0) + suffix;
    detailElement.textContent = detail;
  }

  function relativeSampleTime(value) {
    const when = new Date(value).getTime();
    if (!Number.isFinite(when)) return "No passive sample yet";
    const seconds = Math.max(0, Math.round((Date.now() - when) / 1000));
    if (seconds < 10) return "Updated just now";
    if (seconds < 60) return "Updated " + seconds + "s ago";
    const minutes = Math.round(seconds / 60);
    if (minutes < 60) return "Updated " + minutes + "m ago";
    return "Updated " + Math.round(minutes / 60) + "h ago";
  }

  function healthDescription(summary, health) {
    if (!summary.enabled) return "Passive environmental awareness is disabled in configuration.";
    if (!summary.captured_at) return "SystemSense is waiting for its first passive sample.";
    if (health === "critical") return "One or more current readings need immediate attention.";
    if (health === "degraded") return "SystemSense found pressure, an anomaly, or a device issue worth reviewing.";
    if (health === "good") return "No unusual pressure or hardware issues are visible in the latest sample.";
    return "The current health state could not be derived from the available sensors.";
  }

  function addSystemSignal(text, className) {
    const item = document.createElement("li");
    item.textContent = text;
    if (className) item.classList.add(className);
    systemSignalList.appendChild(item);
  }

  function renderSystemSignals(summary, health) {
    systemSignalList.replaceChildren();
    if (!summary.enabled) {
      addSystemSignal("SystemSense is disabled; no passive telemetry is being presented.", "is-warning");
      return;
    }
    if (!summary.captured_at) {
      addSystemSignal("Waiting for the runtime's first passive sample.", "is-warning");
      return;
    }
    if (summary.throttling_detected) {
      addSystemSignal("Processor performance limiting is currently detected.", "is-critical");
    }
    const deviceProblems = finiteNumber(summary.device_problems) || 0;
    if (deviceProblems > 0) {
      addSystemSignal(deviceProblems + " device" + (deviceProblems === 1 ? " has" : "s have") + " a reported problem.", "is-warning");
    }
    (summary.anomalies || []).slice(0, 3).forEach(function (anomaly) {
      const label = SYSTEM_METRIC_LABELS[anomaly.metric] || anomaly.metric || "A reading";
      const current = finiteNumber(anomaly.current);
      const baseline = finiteNumber(anomaly.baseline_median);
      let text = label + " is " + (anomaly.direction === "low" ? "below" : "above") + " its rolling baseline";
      if (current !== null && baseline !== null) text += " (" + current.toFixed(1) + " vs " + baseline.toFixed(1) + ")";
      addSystemSignal(text + ".", health === "critical" ? "is-critical" : "is-warning");
    });
    (summary.probable_causes || []).slice(0, 3).forEach(function (cause) {
      addSystemSignal(String(cause) + ".", "is-warning");
    });
    if (!systemSignalList.children.length) {
      addSystemSignal("No unusual pressure or active hardware warnings detected.", "is-good");
    }
  }

  function renderSystemProcesses(processes) {
    systemProcessList.replaceChildren();
    const rows = Array.isArray(processes) ? processes.slice(0, 4) : [];
    if (!rows.length) {
      const empty = document.createElement("div");
      empty.className = "system-empty";
      empty.textContent = "No background-process sample is available yet.";
      systemProcessList.appendChild(empty);
      return;
    }
    rows.forEach(function (process) {
      const row = document.createElement("div");
      row.className = "system-process";
      const name = document.createElement("strong");
      name.textContent = process.name || "Unknown process";
      const values = document.createElement("span");
      const cpu = finiteNumber(process.cpu_percent) || 0;
      const ram = finiteNumber(process.ram_mb) || 0;
      values.textContent = cpu.toFixed(0) + "% CPU · " + ram.toFixed(0) + " MB";
      row.appendChild(name);
      row.appendChild(values);
      systemProcessList.appendChild(row);
    });
  }

  function renderSystemSummary(summary) {
    lastSystemSummary = summary || {};
    const health = SYSTEM_HEALTH_LABELS[lastSystemSummary.system_health]
      ? lastSystemSummary.system_health : "unknown";
    const healthLabel = SYSTEM_HEALTH_LABELS[health];
    setSystemHealthClass(systemHealthDot, health);
    setSystemHealthClass(systemHeroDot, health);
    systemToggle.setAttribute("aria-label", "System status: " + healthLabel.toLowerCase());
    systemHealthLabel.textContent = healthLabel;
    systemHealthSummary.textContent = healthDescription(lastSystemSummary, health);
    systemUpdated.textContent = relativeSampleTime(lastSystemSummary.captured_at);

    setSystemMetric(systemCpuValue, systemCpuDetail, lastSystemSummary.cpu_percent, "%", (lastSystemSummary.compute_pressure || "unknown") + " compute pressure", 0);
    setSystemMetric(systemGpuValue, systemGpuDetail, lastSystemSummary.gpu_percent, "%", (lastSystemSummary.compute_pressure || "unknown") + " compute pressure", 0);
    setSystemMetric(systemMemoryValue, systemMemoryDetail, lastSystemSummary.memory_percent, "%", (lastSystemSummary.memory_pressure || "unknown") + " memory pressure", 0);
    setSystemMetric(systemTemperatureValue, systemTemperatureDetail, lastSystemSummary.max_temperature_c, "°", (lastSystemSummary.thermal_state || "unknown") + " thermal state", 0);
    setSystemMetric(systemStorageValue, systemStorageDetail, lastSystemSummary.minimum_volume_free_percent, "%", "Lowest free volume", 0);

    const vram = finiteNumber(lastSystemSummary.vram_used_mb);
    if (vram === null) {
      systemVramValue.textContent = "—";
      systemVramDetail.textContent = "Not reported";
    } else if (vram >= 1024) {
      systemVramValue.textContent = (vram / 1024).toFixed(1) + " GB";
      systemVramDetail.textContent = "Used graphics memory";
    } else {
      systemVramValue.textContent = vram.toFixed(0) + " MB";
      systemVramDetail.textContent = "Used graphics memory";
    }

    const inference = lastSystemSummary.inference || {};
    const speed = finiteNumber(inference.current_tokens_per_second);
    const baseline = finiteNumber(inference.baseline_tokens_per_second);
    const deviation = finiteNumber(inference.deviation_percent);
    if (speed === null) {
      systemInferenceValue.textContent = "Learning baseline";
      systemInferenceDetail.textContent = "No completed inference samples yet";
    } else {
      systemInferenceValue.textContent = speed.toFixed(1) + " tokens/s";
      let detail = baseline === null ? "Current generation speed" : "Typical " + baseline.toFixed(1) + " tokens/s";
      if (deviation !== null) detail += " · " + (deviation >= 0 ? "+" : "") + deviation.toFixed(1) + "%";
      systemInferenceDetail.textContent = detail;
    }

    renderSystemSignals(lastSystemSummary, health);
    renderSystemProcesses(lastSystemSummary.background_resource_contention);
    systemLoading.hidden = true;
    systemContent.hidden = false;
  }

  function renderSystemUnavailable() {
    setSystemHealthClass(systemHealthDot, "unknown");
    setSystemHealthClass(systemHeroDot, "unknown");
    systemToggle.setAttribute("aria-label", "System status unavailable");
    systemHealthLabel.textContent = "Status unavailable";
    systemHealthSummary.textContent = "The local broker could not read the latest SystemSense summary.";
    systemUpdated.textContent = "Will retry automatically";
    systemLoading.hidden = true;
    systemContent.hidden = false;
    systemSignalList.replaceChildren();
    addSystemSignal("Telemetry remains local; no fallback or remote service was used.", "is-warning");
    renderSystemProcesses([]);
  }

  function scheduleSystemRefresh() {
    clearTimeout(systemRefreshTimer);
    systemRefreshTimer = setTimeout(loadSystemSummary, systemPanelOpen ? 15000 : 60000);
  }

  async function loadSystemSummary() {
    if (!BASE_URL || systemRefreshInFlight) return;
    systemRefreshInFlight = true;
    systemRefresh.disabled = true;
    if (systemPanelOpen && !lastSystemSummary) {
      systemLoading.hidden = false;
      systemContent.hidden = true;
    }
    try {
      const data = await api("GET", "/v1/systemsense/summary");
      renderSystemSummary(data.summary || {});
    } catch (e) {
      console.warn("LocalPilot: failed to load SystemSense summary", e);
      renderSystemUnavailable();
    } finally {
      systemRefreshInFlight = false;
      systemRefresh.disabled = false;
      scheduleSystemRefresh();
    }
  }

  function openSystemPanel() {
    closeHistory();
    closeSettings();
    systemPanelOpen = true;
    panelEl.classList.add("is-system-open");
    systemPanel.setAttribute("aria-hidden", "false");
    systemToggle.classList.add("is-active");
    systemToggle.setAttribute("aria-expanded", "true");
    clearTimeout(systemRefreshTimer);
    loadSystemSummary();
    systemClose.focus();
  }

  function closeSystemPanel() {
    if (!systemPanelOpen) return;
    systemPanelOpen = false;
    panelEl.classList.remove("is-system-open");
    systemPanel.setAttribute("aria-hidden", "true");
    systemToggle.classList.remove("is-active");
    systemToggle.setAttribute("aria-expanded", "false");
    scheduleSystemRefresh();
  }

  systemToggle.addEventListener("click", function () {
    if (systemPanelOpen) closeSystemPanel();
    else openSystemPanel();
  });
  systemClose.addEventListener("click", function () { closeSystemPanel(); systemToggle.focus(); });
  systemRefresh.addEventListener("click", function () {
    clearTimeout(systemRefreshTimer);
    loadSystemSummary();
  });

  /* ======================================================================
     Message stream (§2.2/§4) — real data, not scripted demo turns.
     ====================================================================== */
  let activeSessionId = null;
  let lastEventId = 0;
  let lastUserContent = "";
  const knownMessageIds = new Set();
  const messageElementsById = new Map(); // message id -> render state

  function scrollToBottom() { messageStream.scrollTop = messageStream.scrollHeight; }

  function appendInlineMarkdown(container, text) {
    const pattern = /(\*\*[^*\n]+\*\*|`[^`\n]+`)/g;
    let position = 0;
    let match;
    while ((match = pattern.exec(text)) !== null) {
      if (match.index > position) container.appendChild(document.createTextNode(text.slice(position, match.index)));
      const token = match[0];
      const element = document.createElement(token.startsWith("**") ? "strong" : "code");
      element.textContent = token.startsWith("**") ? token.slice(2, -2) : token.slice(1, -1);
      container.appendChild(element);
      position = match.index + token.length;
    }
    if (position < text.length) container.appendChild(document.createTextNode(text.slice(position)));
  }

  function renderSafeMarkdown(target, content) {
    target.replaceChildren();
    let fenced = false;
    String(content || "").split("\n").forEach(function (sourceLine) {
      if (sourceLine.trim().startsWith("```")) { fenced = !fenced; return; }
      const line = document.createElement(fenced ? "pre" : "div");
      if (fenced) {
        const code = document.createElement("code"); code.textContent = sourceLine; line.appendChild(code);
      } else {
        const heading = sourceLine.match(/^\s{0,3}(#{1,6})\s+(.*)$/);
        const bullet = sourceLine.match(/^(\s*)[-*+]\s+(.*)$/);
        if (heading) {
          line.className = "markdown-heading";
          appendInlineMarkdown(line, heading[2]);
        } else if (bullet) {
          line.className = "markdown-bullet";
          appendInlineMarkdown(line, "\u2022 " + bullet[2]);
        } else {
          appendInlineMarkdown(line, sourceLine || "\u00a0");
        }
      }
      target.appendChild(line);
    });
  }

  function renderUserTurn(message) {
    const turn = document.createElement("div");
    turn.className = "turn turn--user";
    const label = document.createElement("span"); label.className = "turn-label"; label.textContent = "You";
    const bubble = document.createElement("div"); bubble.className = "bubble"; bubble.textContent = message.content;
    turn.appendChild(label); turn.appendChild(bubble);
    messageStream.appendChild(turn);
    knownMessageIds.add(message.id);
    lastUserContent = message.content;
    scrollToBottom();
  }

  function renderAssistantTurn(message) {
    const turn = document.createElement("div");
    turn.className = "turn turn--assistant";
    const label = document.createElement("span"); label.className = "turn-label"; label.textContent = "LocalPilot";
    const bubble = document.createElement("div"); bubble.className = "bubble-plain";
    const reveal = document.createElement("span"); reveal.className = "reveal-text";
    bubble.appendChild(reveal);
    let caret = null;
    if (message.status === "streaming") {
      caret = document.createElement("span");
      caret.className = "caret";
      bubble.appendChild(caret);
    }
    reveal.textContent = message.content || "";
    turn.appendChild(label);
    turn.appendChild(bubble);
    messageStream.appendChild(turn);
    knownMessageIds.add(message.id);
    messageElementsById.set(message.id, {
      turnEl: turn,
      bubbleEl: bubble,
      revealEl: reveal,
      caretEl: caret,
      revealedLength: (message.content || "").length,
      targetContent: message.content || "",
      revealTimer: null,
      finalizeAfterReveal: false,
    });
    if (message.status === "complete") renderSafeMarkdown(reveal, message.content || "");
    if (message.status === "failed") renderFailedNotice(message.id, message);
    scrollToBottom();
  }

  function renderFailedNotice(messageId, message) {
    const entry = messageElementsById.get(messageId);
    if (!entry) return;
    const promptAtFailure = lastUserContent;
    const div = document.createElement("div");
    div.className = "failed-notice";
    div.textContent = "LocalPilot's runtime restarted before this answer completed. ";
    const retry = document.createElement("button");
    retry.type = "button";
    retry.textContent = "Retry";
    retry.addEventListener("click", function () {
      if (promptAtFailure) {
        composerInput.value = promptAtFailure;
        composerInput.dispatchEvent(new Event("input"));
      }
      composerInput.focus();
    });
    div.appendChild(retry);
    entry.turnEl.appendChild(div);
  }

  function pacedReveal(messageId) {
    const entry = messageElementsById.get(messageId);
    if (!entry || entry.revealTimer) return;
    function step() {
      if (entry.revealedLength < entry.targetContent.length) {
        entry.revealedLength = Math.min(entry.targetContent.length, entry.revealedLength + 1);
        entry.revealEl.textContent = entry.targetContent.slice(0, entry.revealedLength);
        const level = 0.5 + 0.5 * Math.sin(entry.revealedLength * 0.35);
        avatars.forEach(function (a) { a.ventLevel = level; });
        scrollToBottom();
        entry.revealTimer = setTimeout(step, 28);
      } else {
        entry.revealTimer = null;
        avatars.forEach(function (a) { a.ventLevel = 0; });
        if (entry.finalizeAfterReveal) {
          if (entry.caretEl && entry.caretEl.parentNode) entry.caretEl.remove();
          renderSafeMarkdown(entry.revealEl, entry.targetContent);
        }
      }
    }
    step();
  }

  function updateAssistantContent(messageId, newContent) {
    const entry = messageElementsById.get(messageId);
    if (!entry || newContent == null) return;
    entry.targetContent = newContent;
    setGlobalState("speaking");
    pacedReveal(messageId);
  }

  function finalizeMessage(messageId) {
    const entry = messageElementsById.get(messageId);
    if (!entry) return;
    if (entry.revealedLength >= entry.targetContent.length) {
      if (entry.caretEl && entry.caretEl.parentNode) entry.caretEl.remove();
      renderSafeMarkdown(entry.revealEl, entry.targetContent);
    } else {
      entry.finalizeAfterReveal = true;
    }
  }

  /* ======================================================================
     Tool activity (§5) — the same label map from the design doc, sourced
     from the verified tool registry in tools/__init__.py.
     ====================================================================== */
  const TOOL_LABELS = {
    get_system_summary: "Checking system info",
    get_storage_summary: "Checking disk space",
    get_top_processes: "Checking running processes",
    get_startup_items: "Checking startup items",
    get_active_power_plan: "Checking power settings",
    get_defender_summary: "Checking Defender status",
    get_device_problem_summary: "Checking device issues",
    search_public_web: "Searching the web",
    fetch_public_https: "Searching the web",
    list_repository_tree: "Reading the project",
    read_repository_file: "Reading the project",
    search_repository: "Reading the project",
    get_repository_status: "Checking Git status",
    list_github_pull_requests: "Checking pull requests",
    get_github_pull_request: "Checking pull requests",
    get_github_pull_request_diff: "Checking pull requests",
    list_github_issues: "Checking issues",
    get_github_issue: "Checking issues",
    open_windows_app: "Opening an app",
    open_windows_settings: "Opening Settings",
    set_active_power_plan: "Changing power plan",
    restore_power_plan: "Restoring power plan",
  };
  function labelForTool(name) { return TOOL_LABELS[name] || "Working on it"; }

  let activityRunSteps = [];

  function renderActivitySteps() {
    if (!activityRunSteps.length) return;
    const visible = activityRunSteps.slice(-2);
    const extra = activityRunSteps.length - visible.length;
    activityText.textContent = visible.map(function (s) { return s.label; }).join(" \u00b7 ") +
      (extra > 0 ? " +" + extra + " more" : "");
    activitySteps.textContent = activityRunSteps.map(function (s) { return s.ok === false ? "\u25CB" : "\u25CF"; }).join("");
  }

  function onToolStarted(payload) {
    if (!payload || !payload.tool) return;
    const label = labelForTool(payload.tool);
    activityRunSteps.push({ tool: payload.tool, label: label, ok: null });
    activityStrip.classList.add("is-visible");
    renderActivitySteps();
    previewLine.textContent = label + "\u2026";
    previewLine.classList.add("is-visible");
  }

  function onToolCompleted(payload) {
    if (!payload || !payload.tool) return;
    for (let i = activityRunSteps.length - 1; i >= 0; i--) {
      if (activityRunSteps[i].tool === payload.tool && activityRunSteps[i].ok === null) {
        activityRunSteps[i].ok = !!payload.ok;
        break;
      }
    }
    renderActivitySteps();
  }

  function collapseActivityIntoChip(messageId) {
    activityStrip.classList.remove("is-visible");
    previewLine.classList.remove("is-visible");
    if (!activityRunSteps.length) return;
    const entry = messageElementsById.get(messageId);
    if (entry) {
      const chip = document.createElement("div");
      chip.className = "activity-chip";
      chip.textContent = "\u2699 " + activityRunSteps.length + " step" + (activityRunSteps.length > 1 ? "s" : "");
      chip.title = activityRunSteps.map(function (s) { return s.label; }).join(", ");
      entry.bubbleEl.appendChild(chip);
    }
    activityRunSteps = [];
  }

  /* ======================================================================
     Event dispatcher (§10.1) — verified against the real, live event log.
     Unrecognized event types (raw supervisor passthrough like
     runtime.starting/runtime.exited) are ignored by design, not by
     omission — see delivery notes.
     ====================================================================== */
  function handleEvent(evt) {
    switch (evt.type) {
      case "runtime.ready":
        if (currentState === "offline" || currentState === "restarting") setGlobalState("idle");
        break;
      case "runtime.state": {
        const state = evt.payload && evt.payload.state;
        if (state) setGlobalState(state);
        break;
      }
      case "runtime.error":
        setGlobalState("error");
        break;
      case "session.created":
        if (historySheet.classList.contains("is-open")) loadSessionList();
        break;
      case "message.created": {
        const message = evt.payload && evt.payload.message;
        if (!message || message.session_id !== activeSessionId || knownMessageIds.has(message.id)) break;
        if (message.role === "user") renderUserTurn(message);
        else renderAssistantTurn(message);
        break;
      }
      case "assistant.delta": {
        const messageId = evt.payload && evt.payload.message_id;
        const content = evt.payload && evt.payload.content;
        if (messageId == null || content == null || !messageElementsById.has(messageId)) break;
        updateAssistantContent(messageId, content);
        break;
      }
      case "message.completed": {
        const message = evt.payload && evt.payload.message;
        if (!message || !messageElementsById.has(message.id)) break;
        updateAssistantContent(message.id, message.content);
        finalizeMessage(message.id);
        collapseActivityIntoChip(message.id);
        loadSessionList();
        break;
      }
      case "message.delayed": {
        const seconds = evt.payload && evt.payload.timeout_seconds;
        previewLine.textContent = "Still working" + (seconds ? " after " + seconds + " seconds" : "") + "\u2026";
        previewLine.classList.add("is-visible");
        setGlobalState("working");
        break;
      }
      case "message.failed": {
        const message = evt.payload && evt.payload.message;
        if (!message || !messageElementsById.has(message.id)) break;
        const entry = messageElementsById.get(message.id);
        if (entry.caretEl && entry.caretEl.parentNode) entry.caretEl.remove();
        renderFailedNotice(message.id, message);
        activityStrip.classList.remove("is-visible");
        previewLine.classList.remove("is-visible");
        break;
      }
      case "tool.started":
        onToolStarted(evt.payload);
        break;
      case "tool.completed":
        onToolCompleted(evt.payload);
        break;
      default:
        break; // raw supervisor passthrough types — no UI meaning, ignored on purpose
    }
  }

  /* ======================================================================
     Session loading and switching
     ====================================================================== */
  async function loadMessages(sessionId) {
    const data = await api("GET", "/v1/sessions/" + sessionId + "/messages");
    return data.messages || [];
  }

  async function switchSession(sessionId) {
    activeSessionId = sessionId;
    knownMessageIds.clear();
    messageElementsById.clear();
    messageStream.replaceChildren();
    lastUserContent = "";
    try {
      const messages = await loadMessages(sessionId);
      messages.forEach(function (m) {
        if (m.role === "user") renderUserTurn(m);
        else renderAssistantTurn(m);
      });
      await loadSessionList();
    } catch (e) {
      console.error("LocalPilot: failed to load messages for session", sessionId, e);
    }
  }

  async function bootstrapSession() {
    const sessions = await loadSessionList();
    if (sessions.length) {
      await switchSession(sessions[0].id);
    } else {
      const created = await api("POST", "/v1/sessions", {});
      await switchSession(created.session.id);
    }
  }

  /* ======================================================================
     Long-poll loop (§0/§10) — the broker fakes streaming and has no push
     transport; this recurses on every response, replaying from after=0 on
     the very first call so runtime.state history (and any gap between the
     REST load above and this loop starting) is naturally caught up via the
     same dedup path used for live events.
     ====================================================================== */
  async function pollEvents() {
    try {
      const path = "/v1/events?after=" + lastEventId + "&wait=25" +
        (activeSessionId ? "&session_id=" + encodeURIComponent(activeSessionId) : "");
      const data = await api("GET", path);
      const events = data.events || [];
      events.forEach(function (evt) {
        handleEvent(evt);
        if (evt.id > lastEventId) lastEventId = evt.id;
      });
    } catch (e) {
      await sleep(1500); // broker likely unreachable; health loop will surface "offline"
    }
    pollEvents();
  }

  /* ======================================================================
     Health polling (§10.2) — narrow job: detect offline vs restarting.
     Everything else is owned by the event stream above, so the two
     mechanisms never fight over which state to show.
     ====================================================================== */
  let lastHealthState = null;
  async function healthLoop() {
    const result = await checkHealth();
    if (!result.reachable) {
      if (lastHealthState !== "offline") setGlobalState("offline");
      lastHealthState = "offline";
      settingsStatusText.textContent = "Broker unreachable";
    } else if (result.runtime === "restarting") {
      if (lastHealthState !== "restarting") setGlobalState("restarting");
      lastHealthState = "restarting";
      settingsStatusText.textContent = "Restarting\u2026";
    } else {
      lastHealthState = "running";
      settingsStatusText.textContent = "Runtime running";
    }
    setTimeout(healthLoop, 4000);
  }

  /* ======================================================================
     Composer (§4) — real submission; rendering happens via the event
     stream (message.created), not here, so there is one source of truth
     for what's actually in the conversation.
     ====================================================================== */
  composerInput.addEventListener("focus", function () {
    composerWrap.classList.add("is-focused");
    if (currentState === "idle" || currentState === "sleeping") setGlobalState("listening");
  });
  composerInput.addEventListener("blur", function () {
    composerWrap.classList.remove("is-focused");
    if (currentState === "listening" && !composerInput.value.trim()) setGlobalState("idle");
  });
  composerInput.addEventListener("input", function () {
    sendBtn.classList.toggle("is-ready", composerInput.value.trim().length > 0);
    if (currentState === "idle" && composerInput.value.trim()) setGlobalState("listening");
  });
  composerInput.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); trySend(); }
  });
  sendBtn.addEventListener("click", trySend);

  async function trySend() {
    const text = composerInput.value.trim();
    if (!text || !activeSessionId) return;
    composerInput.value = "";
    sendBtn.classList.remove("is-ready");
    composerWrap.classList.add("is-busy");
    try {
      await api("POST", "/v1/sessions/" + activeSessionId + "/messages", { content: text });
      loadSessionList();
    } catch (e) {
      console.error("LocalPilot: failed to send message", e);
      composerInput.value = text; // don't silently lose what they typed
      composerInput.dispatchEvent(new Event("input"));
    } finally {
      composerWrap.classList.remove("is-busy");
    }
  }

  /* ======================================================================
     Bootstrap — called once by Python (webview_app.py) after window load.
     ====================================================================== */
  window.__initLocalPilot = function (payload) {
    BASE_URL = payload.baseUrl;
    TOKEN = payload.token;
    hasConfigPath = !!payload.hasConfigPath;
    openConfigBtn.hidden = !hasConfigPath;

    bridge("get_start_with_windows").then(function (result) {
      if (result && result.ok) {
        startupToggle.classList.toggle("is-on", !!result.enabled);
        startupToggle.setAttribute("aria-checked", String(!!result.enabled));
      }
    });

    setGlobalState("restarting");
    bootstrapSession()
      .then(function () { pollEvents(); })
      .catch(function (e) { console.error("LocalPilot: failed to bootstrap session", e); });

    loadSystemSummary();
    healthLoop();
    armSleepTimer();
  };

})();
