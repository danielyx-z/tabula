import * as AlphaTab from "/static/alphatab/alphaTab.min.mjs";

(() => {
  "use strict";

  const byId = (id) => document.getElementById(id);

  const form = byId("upload-form");
  const midiInput = byId("midi-file");
  const dropZone = byId("drop-zone");
  const dropTitle = byId("drop-title");
  const dropHint = byId("drop-hint");
  const selectedFile = byId("selected-file");
  const selectedName = byId("selected-name");
  const selectedSize = byId("selected-size");
  const clearFileButton = byId("clear-file");
  const convertButton = byId("convert-button");
  const convertLabel = byId("convert-label");
  const formStatus = byId("form-status");

  const emptyState = byId("empty-state");
  const loadingState = byId("loading-state");
  const errorState = byId("error-state");
  const errorMessage = byId("error-message");
  const errorDetails = byId("error-details");
  const errorLog = byId("error-log");
  const retryButton = byId("retry-button");
  const scoreStage = document.querySelector(".score-stage");

  const scoreView = byId("score-view");
  const alphaTabEl = byId("alphaTab");
  const scoreLoading = byId("score-loading");
  const playButton = byId("play-button");
  const playIcon = byId("play-icon");
  const pauseIcon = byId("pause-icon");
  const currentTimeEl = byId("current-time");
  const totalTimeEl = byId("total-time");
  const seek = byId("seek");
  const barReadout = byId("bar-readout");
  const playerStatus = byId("player-status");

  // Practice controls
  const prevBarButton = byId("prev-bar");
  const nextBarButton = byId("next-bar");
  const loopButton = byId("loop-button");
  const followButton = byId("follow-button");
  const speedSelect = byId("speed-select");
  const muteButton = byId("mute-button");
  const volumeSlider = byId("volume-slider");
  const cancelButton = byId("cancel-button");

  const sessionPanel = byId("session-panel");
  const sessionFile = byId("session-file");
  const factCapo = byId("fact-capo");
  const factNotes = byId("fact-notes");
  const factOpen = byId("fact-open");
  const factMax = byId("fact-max");
  const factHigh = byId("fact-high");
  const factDropped = byId("fact-dropped");
  const downloadGp5 = byId("download-gp5");
  const downloadMidi = byId("download-midi");
  const downloadTab = byId("download-tab");
  const downloadReport = byId("download-report");
  const exportNote = byId("export-note");

  const serverDot = byId("server-dot");
  const serverLabel = byId("server-label");
  const startOverButton = byId("start-over-button");
  const recentList = byId("recent-list");

  const MAX_MB = Number(form.dataset.maxMb) || 32;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  const LayoutMode = AlphaTab && AlphaTab.LayoutMode;
  const ScrollMode = AlphaTab && AlphaTab.ScrollMode;
  const PlayerMode = AlphaTab
    ? AlphaTab.PlayerMode || (AlphaTab.synth && AlphaTab.synth.PlayerMode)
    : null;

  let currentFile = null;
  let lastFileName = "";
  let busy = false;
  let serverReady = false;
  let abortController = null;

  let api = null;
  let score = null;
  let bars = [];
  let scoreReady = false;
  let playerReady = false;
  let audioProblem = "";

  let currentBar = -1;
  let lastTick = 0;
  let lastEndTick = 0;
  let lastEndTime = 0;
  let seeking = false;
  let isPlaying = false;

  // Persisted practice preferences
  let playbackSpeed = Number(localStorage.getItem("tabula:playbackSpeed")) || 1;
  if (!Number.isFinite(playbackSpeed) || playbackSpeed <= 0) playbackSpeed = 1;
  let masterVolume = Number(localStorage.getItem("tabula:masterVolume"));
  if (!Number.isFinite(masterVolume) || masterVolume < 0 || masterVolume > 1) masterVolume = 1;
  let isMuted = localStorage.getItem("tabula:isMuted") === "1";
  let followEnabled = localStorage.getItem("tabula:followEnabled") !== "0"; // on by default
  let loopEnabled = localStorage.getItem("tabula:loopEnabled") === "1";

  const MAX_RECENT = 5;

  function formatTime(milliseconds) {
    if (!Number.isFinite(milliseconds) || milliseconds <= 0) {
      return "0:00";
    }
    const seconds = Math.max(0, Math.round(milliseconds / 1000));
    return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
  }

  function formatBytes(bytes) {
    if (!Number.isFinite(bytes) || bytes < 1024) {
      return `${bytes || 0} B`;
    }
    const units = ["KB", "MB", "GB"];
    let value = bytes / 1024;
    let unit = units[0];
    for (let i = 1; i < units.length && value >= 1024; i += 1) {
      value /= 1024;
      unit = units[i];
    }
    return `${value >= 10 ? Math.round(value) : value.toFixed(1)} ${unit}`;
  }

  function slugify(value) {
    const stem = String(value || "tabula").replace(/\.[^.]+$/, "");
    const safe = stem
      .replace(/[^\p{L}\p{N}._-]+/gu, "_")
      .replace(/^_+|_+$/g, "")
      .slice(0, 64);
    return safe || "tabula";
  }

  function escapeHtml(value) {
    return String(value || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function setStatus(message) {
    formStatus.textContent = message || "";
  }

  function setPlayerStatus(message) {
    playerStatus.textContent = message || "";
  }

  function setState(name) {
    emptyState.hidden = name !== "empty";
    loadingState.hidden = name !== "loading";
    errorState.hidden = name !== "error";
    scoreView.hidden = name !== "score";
    scoreStage.setAttribute("aria-busy", name === "loading" ? "true" : "false");
    scoreStage.dataset.state = name;
  }

  function setServerState(ok) {
    serverReady = Boolean(ok);
    serverDot.dataset.state = ok ? "ok" : "down";
    serverLabel.textContent = ok ? "local server" : "unavailable";
    if (!busy) {
      convertButton.disabled = !currentFile || !serverReady;
    }
    if (!ok) {
      setStatus("The server isn't responding. Is the Tabula workbench running?");
    }
  }

  function setBusy(value) {
    busy = value;
    convertButton.classList.toggle("is-loading", value);
    convertLabel.textContent = value ? "Making tab" : "Make tab";
    convertButton.disabled = value || !currentFile || !serverReady;
    midiInput.disabled = value;
    clearFileButton.disabled = value;
    cancelButton.disabled = !value;
    form.setAttribute("aria-busy", value ? "true" : "false");
    dropZone.setAttribute("aria-disabled", value ? "true" : "false");
    if (value) {
      dropZone.classList.remove("is-dragging");
      abortController = new AbortController();
    } else {
      abortController = null;
    }
    updatePlayAvailability();
  }

  function chooseFile(file) {
    if (!file || busy) {
      return;
    }
    const name = file.name || "upload.mid";
    if (!/\.(mid|midi)$/i.test(name)) {
      setStatus("Choose a .mid or .midi file.");
      return;
    }
    if (file.size === 0) {
      setStatus("That file is empty.");
      return;
    }
    if (file.size > MAX_MB * 1024 * 1024) {
      setStatus(`That file is larger than ${MAX_MB} MB.`);
      return;
    }

    currentFile = file;
    setStatus("");
    selectedName.textContent = name;
    selectedName.title = name;
    selectedSize.textContent = formatBytes(file.size);
    selectedFile.hidden = false;
    dropTitle.textContent = "Replace the MIDI";
    dropHint.textContent = `drop another file, or browse · ${MAX_MB} MB max`;
    dropZone.classList.add("has-file");
    convertButton.disabled = !serverReady;
    convertLabel.textContent = lastConverted ? "Convert again" : "Make tab";

    if (!errorState.hidden) {
      setState("empty");
    }
  }

  let lastConverted = false;

  function clearFile() {
    if (busy) {
      return;
    }
    currentFile = null;
    midiInput.value = "";
    selectedFile.hidden = true;
    selectedName.textContent = "";
    selectedName.title = "";
    selectedSize.textContent = "";
    dropTitle.textContent = "Drop a MIDI here";
    dropHint.textContent = `or browse · ${MAX_MB} MB max`;
    dropZone.classList.remove("has-file");
    convertButton.disabled = true;
    setStatus("");
    if (!errorState.hidden) {
      setState("empty");
    }
  }

  function showError(message, detail) {
    setState("error");
    errorMessage.textContent = message || "Something went wrong.";
    if (detail) {
      errorLog.textContent = detail;
      errorDetails.hidden = false;
      errorDetails.open = false;
    } else {
      errorLog.textContent = "";
      errorDetails.hidden = true;
    }
  }

  function updatePlayAvailability() {
    if (!api || !scoreReady) {
      playButton.disabled = true;
      seek.disabled = true;
      prevBarButton.disabled = true;
      nextBarButton.disabled = true;
      speedSelect.disabled = true;
      muteButton.disabled = true;
      volumeSlider.disabled = true;
      if (!scoreReady) {
        setPlayerStatus("");
      }
      return;
    }

    if (audioProblem) {
      playButton.disabled = true;
      seek.disabled = true;
      prevBarButton.disabled = true;
      nextBarButton.disabled = true;
      speedSelect.disabled = true;
      muteButton.disabled = true;
      volumeSlider.disabled = true;
      setPlayerStatus(audioProblem);
      return;
    }

    let playable = false;
    try {
      playable = api.isReadyForPlayback === true;
    } catch (error) {
      playable = false;
    }

    playButton.disabled = !playable;
    seek.disabled = !playable;
    prevBarButton.disabled = !playable || !bars.length;
    nextBarButton.disabled = !playable || !bars.length;
    speedSelect.disabled = !playable;
    muteButton.disabled = !playable;
    volumeSlider.disabled = !playable || isMuted;

    if (playable) {
      setPlayerStatus("");
    } else if (playerReady) {
      setPlayerStatus("Preparing audio…");
    } else {
      setPlayerStatus("Loading the player…");
    }
  }

  function applyPlaybackSpeed() {
    if (!api) return;
    try {
      api.player.playbackSpeed = playbackSpeed;
    } catch (error) {
      // Some builds clamp or ignore; ignore gracefully.
    }
    speedSelect.value = String(playbackSpeed);
    try { localStorage.setItem("tabula:playbackSpeed", String(playbackSpeed)); } catch {}
  }

  function applyVolume() {
    if (!api) return;
    try {
      api.player.masterVolume = isMuted ? 0 : masterVolume;
    } catch (error) {
      // ignore
    }
    volumeSlider.disabled = isMuted;
    volumeSlider.value = isMuted ? 0 : masterVolume;
    muteButton.setAttribute("aria-pressed", isMuted ? "true" : "false");
    muteButton.classList.toggle("is-muted", isMuted);
    try {
      localStorage.setItem("tabula:masterVolume", String(masterVolume));
      localStorage.setItem("tabula:isMuted", isMuted ? "1" : "0");
    } catch {}
  }

  function toggleMute() {
    isMuted = !isMuted;
    applyVolume();
    updatePlayAvailability();
  }

  function setFollow(enabled) {
    followEnabled = enabled;
    followButton.setAttribute("aria-pressed", followEnabled ? "true" : "false");
    followButton.classList.toggle("is-on", followEnabled);
    try { localStorage.setItem("tabula:followEnabled", followEnabled ? "1" : "0"); } catch {}
  }

  function setLoop(enabled) {
    loopEnabled = enabled;
    loopButton.setAttribute("aria-pressed", loopEnabled ? "true" : "false");
    loopButton.classList.toggle("is-on", loopEnabled);
    try { localStorage.setItem("tabula:loopEnabled", loopEnabled ? "1" : "0"); } catch {}
  }

  function setPlaying(playing) {
    isPlaying = Boolean(playing);
    playIcon.toggleAttribute("hidden", isPlaying);
    pauseIcon.toggleAttribute("hidden", !isPlaying);
    playButton.setAttribute("aria-label", isPlaying ? "Pause" : "Play");
    playButton.classList.toggle("is-playing", isPlaying);
  }

  function initAlphaTab() {
    if (!AlphaTab || typeof AlphaTab.AlphaTabApi !== "function") {
      showError(
        "The tab renderer did not load.",
        "The local alphaTab bundle could not be found. Check webapp/static/alphatab/."
      );
      return;
    }

    const settings = {
      core: {
        scriptFile: "/static/alphatab/alphaTab.worker.min.mjs",
        fontDirectory: "/static/alphatab/font/",
      },
      display: {
        layoutMode: LayoutMode ? LayoutMode.Parchment : 2,
        scale: 0.98,
      },
      player: {
        playerMode: PlayerMode ? PlayerMode.EnabledSynthesizer : 2,
        soundFont: "/static/alphatab/soundfont/sonivox.sf2",
        scrollElement: alphaTabEl,
        scrollMode: ScrollMode ? ScrollMode.Off : 0,
        enableCursor: true,
        enableAnimatedBeatCursor: true,
        enableElementHighlighting: false,
        enableUserInteraction: true,
        playbackRangeStartTime: 0,
        playbackRangeEndTime: lastEndTick > 0 ? lastEndTick : 0,
      },
    };

    api = new AlphaTab.AlphaTabApi(alphaTabEl, settings);

    api.scoreLoaded.on((loadedScore) => {
      score = loadedScore;
      bars = (loadedScore && loadedScore.masterBars) || [];
      scoreReady = true;
      currentBar = -1;
      lastTick = 0;
      const lastBar = bars[bars.length - 1];
      lastEndTick = lastBar ? lastBar.start + lastBar.calculateDuration() : 0;
      barReadout.textContent = bars.length
        ? `Bar 1 / ${bars.length}`
        : "Bar —";
      updatePlayAvailability();
      applyPlaybackSpeed();
      applyVolume();
    });

    api.renderFinished.on(() => {
      scoreLoading.hidden = true;
      updateMeasure(lastTick, true);
      updatePlayAvailability();
    });

    api.playerReady.on(() => {
      playerReady = true;
      updatePlayAvailability();
    });

    api.soundFontLoaded.on(() => {
      updatePlayAvailability();
    });

    api.playerStateChanged.on(() => {
      updatePlayAvailability();
    });

    api.playerFinished.on(() => {
      setPlaying(false);
      if (loopEnabled && bars.length && currentBar >= 0) {
        // Restart the current bar from its start tick.
        const bar = bars[currentBar];
        const startTick = bar.start;
        try { api.tickPosition = startTick; } catch {}
        lastTick = startTick;
        currentTimeEl.textContent = formatTime((startTick / Math.max(1, lastEndTick)) * lastEndTime);
        if (!isPlaying) {
          try { api.play(); } catch {}
          setPlaying(true);
        }
      }
    });

    api.playerPositionChanged.on((args) => {
      const tick = Number(args.currentTick) || 0;
      const endTick = Number(args.endTick) || 0;
      const currentTime = Number(args.currentTime) || 0;
      const endTime = Number(args.endTime) || 0;

      lastTick = tick;
      if (endTick > 0) {
        lastEndTick = endTick;
      }
      if (endTime > 0) {
        lastEndTime = endTime;
        totalTimeEl.textContent = formatTime(endTime);
      }

      if (!seeking) {
        currentTimeEl.textContent = formatTime(currentTime);
        if (lastEndTick > 0) {
          seek.value = String(Math.max(0, Math.min(1000, Math.round((tick / lastEndTick) * 1000))));
        }
      }
      updateMeasure(tick, false, followEnabled);
    });

    api.error.on((error) => {
      const message = error && error.message ? error.message : String(error || "Unknown error");
      if (!scoreReady) {
        showError("The tab could not be rendered.", message);
        return;
      }
      audioProblem = "Audio is unavailable; the tab is still viewable.";
      playButton.disabled = true;
      seek.disabled = true;
      setPlayerStatus(audioProblem);
    });
  }

  function barIndexAt(tick) {
    if (!bars.length) {
      return -1;
    }
    let index = 0;
    for (let i = 0; i < bars.length; i += 1) {
      if (tick >= bars[i].start) {
        index = i;
      } else {
        break;
      }
    }
    return index;
  }

  function barStartTick(index) {
    return bars[index] ? bars[index].start : 0;
  }

  function updateMeasure(tick, force = false, reveal = false) {
    if (!scoreReady || !api || !bars.length) {
      return;
    }
    const index = barIndexAt(tick);
    if (index < 0) {
      return;
    }
    if (!force && index === currentBar) {
      return;
    }
    currentBar = index;
    barReadout.textContent = `Bar ${index + 1} / ${bars.length}`;
    if (reveal && followEnabled) {
      scrollMeasureIntoView(index);
    }
  }

  function scrollMeasureIntoView(index) {
    if (!api || !api.renderer || !api.renderer.boundsLookup) {
      return;
    }
    const bar = api.renderer.boundsLookup.findMasterBarByIndex(index);
    if (!bar || !(bar.visualBounds.h > 0)) {
      return;
    }
    const y = bar.visualBounds.y;
    const top = alphaTabEl.scrollTop;
    const bottom = top + alphaTabEl.clientHeight;
    if (y < top + 48 || y + bar.visualBounds.h > bottom - 24) {
      alphaTabEl.scrollTop = Math.max(0, y - 72);
    }
  }

  function setFact(element, value, title) {
    element.textContent = value;
    if (title) {
      element.title = title;
    } else {
      element.removeAttribute("title");
    }
  }

  function percent(rate, count, total) {
    let value = Number(rate);
    if (!Number.isFinite(value)) {
      value = total > 0 ? count / total : 0;
    }
    return `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;
  }

  // ---------- Recent takes (localStorage) ----------

  function getRecent() {
    try {
      return JSON.parse(localStorage.getItem("tabula:recent") || "[]");
    } catch {
      return [];
    }
  }

  function setRecent(arr) {
    try {
      localStorage.setItem("tabula:recent", JSON.stringify(arr.slice(0, MAX_RECENT)));
    } catch {
      // storage unavailable; recent takes won't persist
    }
  }

  function renderRecent() {
    const arr = getRecent();
    recentList.hidden = arr.length === 0;
    recentList.innerHTML = arr
      .map(
        (r) =>
          `<button type="button" class="recent-item" data-id="${escapeHtml(
            r.id
          )}" title="${escapeHtml(r.name)}" aria-label="Open ${escapeHtml(r.name)}">
            <span class="recent-name">${escapeHtml(r.name)}</span>
            <span class="recent-meta">${r.notes} notes · ${formatTime(
            (r.duration || 0) * 1000
          )}</span>
          </button>`
      )
      .join("");
    recentList.querySelectorAll(".recent-item").forEach((btn) => {
      btn.addEventListener("click", () => {
        const id = btn.dataset.id;
        const entry = getRecent().find((r) => r.id === id);
        if (!entry) return;
        restoreRecent(entry);
      });
    });
  }

  function addRecent(entry) {
    let arr = getRecent().filter((r) => r.id !== entry.id);
    arr.unshift(entry);
    setRecent(arr);
    renderRecent();
  }

  function restoreRecent(entry) {
    if (busy) return;
    const payload = {
      urls: entry.urls,
      summary: {
        notes: entry.notes || 0,
        capo: entry.capo || 0,
        open_notes: entry.open_notes || 0,
        high_notes: entry.high_notes || 0,
        dropped: entry.dropped || 0,
        exported: { duration: entry.duration || 0, skipped: 0 },
      },
      title: entry.name,
      elapsed: 0,
    };
    lastFileName = entry.name;
    lastConverted = true;
    populateSession(payload, entry.name);
    setState("score");
    loadScore(payload);
  }

  // ---------- End recent takes ----------

  function populateSession(payload, fileName) {
    const summary = payload.summary || {};
    const exported = summary.exported || {};
    const notes = Number(summary.notes) || 0;
    const openNotes = Number(summary.open_notes) || 0;
    const highNotes = Number(summary.high_notes) || 0;
    const dropped = Number(summary.dropped) || 0;
    const capo = Number(summary.capo) || 0;

    sessionFile.textContent = fileName || payload.title || "Uploaded MIDI";
    sessionFile.title = sessionFile.textContent;

    setFact(factCapo, capo > 0 ? `Fret ${capo}` : "None");
    setFact(factNotes, notes ? notes.toLocaleString() : "—");
    setFact(
      factOpen,
      notes ? `${openNotes.toLocaleString()} · ${percent(summary.open_rate, openNotes, notes)}` : "—",
      "Notes played on open strings after the capo."
    );
    setFact(factMax, Number.isFinite(Number(summary.max_fret)) ? String(summary.max_fret) : "—");
    setFact(
      factHigh,
      notes ? `${highNotes.toLocaleString()} · ${percent(summary.high_rate, highNotes, notes)}` : "—",
      "Notes above fret 6. The model avoids them when a lower position is musically sound."
    );
    setFact(
      factDropped,
      `${dropped.toLocaleString()}`,
      dropped
        ? "Inner chord tones removed to keep the voicing playable."
        : "No chord tones were dropped."
    );

    const duration = Number(exported.duration);
    lastEndTime = Number.isFinite(duration) && duration > 0 ? duration * 1000 : 0;
    totalTimeEl.textContent = formatTime(lastEndTime);
    currentTimeEl.textContent = "0:00";
    seek.value = "0";

    const slug = slugify(fileName || payload.title || "tabula");
    downloadGp5.href = payload.urls.gp5;
    downloadGp5.setAttribute("download", `${slug}.gp5`);
    downloadMidi.href = payload.urls.midi;
    downloadMidi.setAttribute("download", `${slug}.mid`);
    downloadTab.href = payload.urls.tab;
    downloadTab.setAttribute("download", `${slug}_tab.txt`);
    downloadReport.href = payload.urls.report;
    downloadReport.setAttribute("download", `${slug}_report.json`);

    const notesOut = [];
    const skipped = Number(exported.skipped) || 0;
    if (skipped > 0) {
      notesOut.push(
        `${skipped} note${skipped === 1 ? "" : "s"} could not be placed in the Guitar Pro file; the text tab is complete.`
      );
    }
    const elapsed = Number(payload.elapsed);
    if (Number.isFinite(elapsed) && elapsed > 0) {
      const shown = elapsed >= 10 ? Math.round(elapsed) : elapsed.toFixed(1);
      notesOut.push(`Converted in ${shown}s.`);
    }
    exportNote.textContent = notesOut.join(" ");
    exportNote.hidden = notesOut.length === 0;

    sessionPanel.hidden = false;
  }

  function loadScore(payload) {
    score = null;
    bars = [];
    scoreReady = false;
    currentBar = -1;

    if (api) {
      try {
        api.stop();
      } catch (error) {
        // A player that has not loaded a score yet can throw on stop.
      }
    }
    setPlaying(false);
    sessionPanel.hidden = true;
    setBusy(true);

    currentBar = -1;
    lastTick = 0;
    lastEndTick = 0;
    lastEndTime = 0;
    audioProblem = "";
    barReadout.textContent = "Bar —";
    setPlaying(false);
    updatePlayAvailability();
    scoreLoading.hidden = false;
    setPlayerStatus("Laying out the tab…");

    let loaded = false;
    try {
      loaded = api.load(payload.urls.gp5);
    } catch (error) {
      showError("The tab could not be opened.", error && error.message ? error.message : String(error));
      setBusy(false);
      return;
    }
    if (!loaded) {
      showError(
        "The tab could not be opened.",
        "alphaTab refused to load the generated Guitar Pro file."
      );
      setBusy(false);
    }
  }

  function handleConverted(payload) {
    const fileName = lastFileName || (currentFile && currentFile.name) || "";
    populateSession(payload, fileName);
    setState("score");
    loadScore(payload);
    lastConverted = true;

    // Remember the take for recent-takes restore.
    const summary = payload.summary || {};
    const exported = summary.exported || {};
    addRecent({
      id: payload.job_id || slugify(fileName),
      name: fileName || payload.title || "tabula",
      date: Date.now(),
      duration: Number(exported.duration) || 0,
      notes: Number(summary.notes) || 0,
      capo: Number(summary.capo) || 0,
      open_notes: Number(summary.open_notes) || 0,
      high_notes: Number(summary.high_notes) || 0,
      dropped: Number(summary.dropped) || 0,
      urls: {
        gp5: payload.urls.gp5,
        midi: payload.urls.midi,
        tab: payload.urls.tab,
        report: payload.urls.report,
      },
    });

    if (window.matchMedia("(max-width: 980px)").matches) {
      window.requestAnimationFrame(() => {
        scoreStage.scrollIntoView({
          behavior: reduceMotion.matches ? "auto" : "smooth",
          block: "start",
        });
      });
    }
  }

  async function convertFile() {
    if (!currentFile || busy || !serverReady) {
      return;
    }

    lastFileName = currentFile.name;
    setStatus("");
    if (api) {
      try {
        api.stop();
      } catch (error) {
        // A player that has not loaded a score yet can throw on stop.
      }
    }
    setPlaying(false);
    sessionPanel.hidden = true;
    setState("loading");
    setBusy(true);

    const body = new FormData();
    body.append("file", currentFile, currentFile.name);

    const signal = abortController && abortController.signal;

    try {
      const response = await fetch("/api/convert", {
        method: "POST",
        body,
        headers: { Accept: "application/json" },
        signal,
      });

      if (abortController && abortController.signal.aborted) {
        return;
      }

      let data = null;
      try {
        data = await response.json();
      } catch (error) {
        data = null;
      }

      if (!response.ok) {
        const message = (data && data.error) || `The converter returned HTTP ${response.status}.`;
        const failure = new Error(message);
        failure.detail = (data && data.detail) || "";
        throw failure;
      }
      if (!data || !data.urls || !data.urls.gp5) {
        throw new Error("The converter returned an incomplete result.");
      }

      handleConverted(data);
    } catch (error) {
      if (error.name === "AbortError") {
        setStatus("Conversion cancelled.");
        setState("empty");
        return;
      }
      showError(
        (error && error.message) || "The conversion failed.",
        (error && error.detail) || ""
      );
    } finally {
      setBusy(false);
    }
  }

  function cancelConversion() {
    if (abortController) {
      abortController.abort();
    }
  }

  function checkHealth() {
    fetch("/api/health", { cache: "no-store", headers: { Accept: "application/json" } })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        return response.json();
      })
      .then((data) => {
        const ok = Boolean(data.ok && data.checkpoint_ready && data.vocab_ready);
        setServerState(ok);
        if (data.ok && !(data.checkpoint_ready && data.vocab_ready)) {
          setStatus("The model checkpoint or vocabulary is missing on this PC.");
        }
      })
      .catch(() => {
        setServerState(false);
      });
  }

  function startOver() {
    if (busy) return;
    if (api) {
      try {
        api.stop();
      } catch {}
    }
    setPlaying(false);
    currentFile = null;
    midiInput.value = "";
    selectedFile.hidden = true;
    selectedName.textContent = "";
    selectedSize.textContent = "";
    dropTitle.textContent = "Drop a MIDI here";
    dropHint.textContent = `or browse · ${MAX_MB} MB max`;
    dropZone.classList.remove("has-file");
    convertButton.disabled = true;
    convertLabel.textContent = "Make tab";
    lastConverted = false;
    setStatus("");
    setState("empty");
  }

  function seekBySeconds(delta) {
    if (!api || !scoreReady || !lastEndTick) return;
    seeking = true;
    const fraction = Number(seek.value) / 1000;
    const newTick = Math.round(((fraction * lastEndTick + delta) / lastEndTick) * 1000);
    seek.value = String(Math.max(0, Math.min(1000, newTick)));
    const tick = Math.round((Number(seek.value) / 1000) * lastEndTick);
    currentTimeEl.textContent = formatTime((tick / Math.max(1, lastEndTick)) * lastEndTime);
    updateMeasure(tick, true);
    window.setTimeout(() => {
      if (!seeking) return;
      seeking = false;
      try { api.tickPosition = tick; } catch {}
      if (followEnabled) scrollMeasureIntoView(barIndexAt(tick));
    }, 120);
  }

  function goToBar(offset) {
    if (!api || !scoreReady || !bars.length) return;
    let target = currentBar + offset;
    target = Math.max(0, Math.min(bars.length - 1, target));
    const tick = barStartTick(target);
    seeking = true;
    seek.value = String(Math.max(0, Math.min(1000, Math.round((tick / Math.max(1, lastEndTick)) * 1000))));
    currentTimeEl.textContent = formatTime((tick / Math.max(1, lastEndTick)) * lastEndTime);
    updateMeasure(tick, true);
    window.setTimeout(() => {
      seeking = false;
      try { api.tickPosition = tick; } catch {}
      if (followEnabled) scrollMeasureIntoView(target);
    }, 120);
  }

  function isInteractiveInput(element) {
    if (!element) return false;
    const tag = element.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
    if (element.isContentEditable) return true;
    return false;
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    convertFile();
  });

  midiInput.addEventListener("change", () => {
    const file = midiInput.files && midiInput.files[0];
    if (file) {
      chooseFile(file);
    }
    midiInput.value = "";
  });

  clearFileButton.addEventListener("click", clearFile);

  ["dragenter", "dragover"].forEach((type) => {
    dropZone.addEventListener(type, (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (!busy) {
        dropZone.classList.add("is-dragging");
      }
    });
  });

  ["dragleave", "dragend"].forEach((type) => {
    dropZone.addEventListener(type, (event) => {
      event.preventDefault();
      event.stopPropagation();
      dropZone.classList.remove("is-dragging");
    });
  });

  dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropZone.classList.remove("is-dragging");
    if (busy) {
      return;
    }
    const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) {
      chooseFile(file);
    }
  });

  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("drop", (event) => event.preventDefault());

  retryButton.addEventListener("click", () => {
    setState("empty");
    midiInput.click();
  });

  startOverButton.addEventListener("click", startOver);

  cancelButton.addEventListener("click", cancelConversion);

  playButton.addEventListener("click", () => {
    if (!api || !scoreReady) {
      return;
    }
    if (!api.isReadyForPlayback) {
      setPlayerStatus("Preparing audio…");
      return;
    }
    if (isPlaying) {
      api.pause();
      setPlaying(false);
      return;
    }
    if (lastEndTick > 0 && lastTick >= lastEndTick - 1) {
      api.tickPosition = bars.length ? bars[0].start : 0;
    }
    applyPlaybackSpeed();
    api.play();
    setPlaying(true);
  });

  prevBarButton.addEventListener("click", () => goToBar(-1));
  nextBarButton.addEventListener("click", () => goToBar(1));

  loopButton.addEventListener("click", () => setLoop(!loopEnabled));
  followButton.addEventListener("click", () => setFollow(!followEnabled));

  speedSelect.addEventListener("change", () => {
    playbackSpeed = Number(speedSelect.value) || 1;
    applyPlaybackSpeed();
  });

  muteButton.addEventListener("click", toggleMute);

  volumeSlider.addEventListener("input", () => {
    isMuted = false;
    masterVolume = Number(volumeSlider.value);
    applyVolume();
  });

  seek.addEventListener("input", () => {
    if (!api || !scoreReady || !lastEndTick) {
      return;
    }
    seeking = true;
    const fraction = Number(seek.value) / 1000;
    const tick = Math.round(fraction * lastEndTick);
    currentTimeEl.textContent = formatTime(fraction * lastEndTime);
    updateMeasure(tick, true);
  });

  const commitSeek = () => {
    if (!api || !scoreReady || !lastEndTick) {
      seeking = false;
      return;
    }
    const fraction = Number(seek.value) / 1000;
    const tick = Math.round(fraction * lastEndTick);
    try {
      api.tickPosition = tick;
    } catch (error) {
      // Seeking is best effort; some player builds ignore it until ready.
    }
    window.setTimeout(() => scrollMeasureIntoView(barIndexAt(tick)), 120);
    seeking = false;
  };

  seek.addEventListener("change", commitSeek);
  seek.addEventListener("pointerup", commitSeek);
  seek.addEventListener("blur", () => {
    if (seeking) {
      commitSeek();
    }
  });

  window.addEventListener("keydown", (event) => {
    if (isInteractiveInput(document.activeElement)) return;
    const key = event.key.toLowerCase();
    switch (key) {
      case " ":
        event.preventDefault();
        playButton.click();
        break;
      case "arrowleft":
        event.preventDefault();
        seekBySeconds(-4);
        break;
      case "arrowright":
        event.preventDefault();
        seekBySeconds(4);
        break;
      case "[":
        event.preventDefault();
        playbackSpeed = Math.max(0.5, playbackSpeed - 0.25);
        speedSelect.value = String(playbackSpeed);
        applyPlaybackSpeed();
        break;
      case "]":
        event.preventDefault();
        playbackSpeed = Math.min(2, playbackSpeed + 0.25);
        speedSelect.value = String(playbackSpeed);
        applyPlaybackSpeed();
        break;
      case "m":
        event.preventDefault();
        muteButton.click();
        break;
      case "f":
        event.preventDefault();
        followButton.click();
        break;
      case "l":
        event.preventDefault();
        loopButton.click();
        break;
    }
  });

  const params = new URLSearchParams(location.search);
  const jobParam = params.get("job");
  if (jobParam) {
    setState("loading");
    loadJobFromUrl(jobParam, params).catch((error) => {
      showError("Could not load job.", String(error));
      setState("empty");
    });
  } else {
    setState("empty");
  }
  initAlphaTab();
  checkHealth();
  renderRecent();
})();

async function loadJobFromUrl(jobParam, params) {
  try {
    const reportRes = await fetch(`/api/jobs/${jobId}/report.json`);
    const report = reportRes.ok ? await reportRes.json() : {};
    const summary = report.summary || {};
    const exported = summary.exported || {};
    const duration = Number(exported.duration) || 0;
    const notes = Number(summary.notes) || 0;

    const payload = {
      job_id: jobId,
      title: `Job ${jobId.slice(0, 8)}`,
      elapsed: 0,
      summary: {
        notes,
        capo: Number(summary.capo) || 0,
        open_notes: Number(summary.open_notes) || 0,
        high_notes: Number(summary.high_notes) || 0,
        dropped: Number(summary.dropped) || 0,
        exported: { duration, skipped: Number(exported.skipped) || 0 },
      },
      urls: {
        gp5: `/api/jobs/${jobId}/score.gp5`,
        midi: `/api/jobs/${jobId}/upload.mid`,
        tab: `/api/jobs/${jobId}/final_tab.txt`,
        report: `/api/jobs/${jobId}/report.json`,
      },
    };

    lastFileName = payload.title;
    lastConverted = true;
    populateSession(payload, payload.title);
    setState("score");
    loadScore(payload);

    const playParam = params.get("play");
    const seekParam = params.get("seek");
    if (playParam === "1") {
      const ready = setInterval(() => {
        if (api && scoreReady && api.isReadyForPlayback) {
          clearInterval(ready);
          if (seekParam) {
            const frac = Math.max(0, Math.min(1, Number(seekParam)));
            seek.value = String(Math.round(frac * 1000));
            const tick = Math.round(frac * lastEndTick);
            currentTimeEl.textContent = formatTime(frac * lastEndTime);
            updateMeasure(tick, true);
          }
          applyPlaybackSpeed();
          api.play();
          setPlaying(true);
        }
      }, 200);
    }
  } catch (error) {
    showError("Could not load job.", String(error));
    setState("empty");
  }
}
