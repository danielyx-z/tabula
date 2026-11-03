import * as AlphaTab from "/static/alphatab/alphaTab.min.mjs";

(() => {
  "use strict";

  const byId = (id) => document.getElementById(id);

  const form = byId("upload-form");
  const uploadDetails = byId("upload-details");
  const audioInput = byId("audio-file");
  const midiInput = byId("midi-file");
  const audioRow = byId("audio-row");
  const midiRow = byId("midi-row");
  const audioTitle = byId("audio-title");
  const midiTitle = byId("midi-title");
  const audioMeta = byId("audio-meta");
  const midiMeta = byId("midi-meta");
  const audioAction = byId("audio-action");
  const midiAction = byId("midi-action");
  const audioClear = byId("audio-clear");
  const midiClear = byId("midi-clear");
  const convertButton = byId("convert-button");
  const convertLabel = byId("convert-label");
  const cancelButton = byId("cancel-button");
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

  const sessionPanel = byId("session-panel");
  const sessionFile = byId("session-file");
  const sessionSource = byId("session-source");
  const factCapo = byId("fact-capo");
  const factNotes = byId("fact-notes");
  const factDead = byId("fact-dead");
  const factHp = byId("fact-hp");
  const factOpen = byId("fact-open");
  const factMax = byId("fact-max");
  const factHigh = byId("fact-high");
  const factDropped = byId("fact-dropped");
  const factMatch = byId("fact-match");
  const downloadGp5 = byId("download-gp5");
  const downloadMidi = byId("download-midi");
  const downloadTab = byId("download-tab");
  const downloadAudio = byId("download-audio");
  const downloadReport = byId("download-report");
  const exportNote = byId("export-note");

  const serverDot = byId("server-dot");
  const serverLabel = byId("server-label");
  const startOverButton = byId("start-over-button");
  const recentSection = byId("recent-takes");
  const recentList = byId("recent-list");

  const MAX_AUDIO_MB = Number(form.dataset.maxAudioMb) || 96;
  const MAX_MIDI_MB = Number(form.dataset.maxMidiMb) || 8;
  const MAX_AUDIO_BYTES = MAX_AUDIO_MB * 1024 * 1024;
  const MAX_MIDI_BYTES = MAX_MIDI_MB * 1024 * 1024;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  const AUDIO_NAME = /\.(wav|flac|ogg|oga|mp3|m4a|aac|aif|aiff|opus|wma)$/i;
  const MIDI_NAME = /\.(mid|midi)$/i;

  const LayoutMode = AlphaTab && AlphaTab.LayoutMode;
  const ScrollMode = AlphaTab && AlphaTab.ScrollMode;
  const PlayerMode = AlphaTab
    ? AlphaTab.PlayerMode || (AlphaTab.synth && AlphaTab.synth.PlayerMode)
    : null;

  let busy = false;
  let serverReady = false;
  let abortController = null;
  let lastConverted = false;
  let lastFileName = "";

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

  // Persisted practice preferences. The old volume key defaulted to 0 because
  // Number(null) is 0, which muted the synth on every first visit.
  let playbackSpeed = Number(localStorage.getItem("tabula:playbackSpeed"));
  if (!Number.isFinite(playbackSpeed) || playbackSpeed <= 0) playbackSpeed = 1;
  const storedVolume = localStorage.getItem("tabula:masterVolume.v2");
  let masterVolume = storedVolume === null ? 1 : Number(storedVolume);
  if (!Number.isFinite(masterVolume) || masterVolume < 0 || masterVolume > 1) {
    masterVolume = 1;
  }
  let isMuted = localStorage.getItem("tabula:isMuted") === "1";
  let followEnabled = localStorage.getItem("tabula:followEnabled") !== "0"; // on by default
  let loopEnabled = localStorage.getItem("tabula:loopEnabled") === "1";

  const MAX_RECENT = 5;

  const slots = {
    audio: {
      kind: "audio",
      input: audioInput,
      row: audioRow,
      title: audioTitle,
      meta: audioMeta,
      action: audioAction,
      clear: audioClear,
      file: null,
      defaultTitle: "Add the recording",
      defaultMeta: `WAV, FLAC, OGG, MP3 · ${MAX_AUDIO_MB} MB`,
      defaultAction: "Browse",
    },
    midi: {
      kind: "midi",
      input: midiInput,
      row: midiRow,
      title: midiTitle,
      meta: midiMeta,
      action: midiAction,
      clear: midiClear,
      file: null,
      defaultTitle: "Add the MIDI",
      defaultMeta: `.mid or .midi · ${MAX_MIDI_MB} MB`,
      defaultAction: "Browse",
    },
  };

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

  function setStatus(message, kind = "error") {
    formStatus.textContent = message || "";
    if (message) {
      formStatus.dataset.kind = kind;
    } else {
      delete formStatus.dataset.kind;
    }
  }

  function setPlayerStatus(message) {
    playerStatus.textContent = message || "";
  }

  function setState(name) {
    emptyState.hidden = name !== "empty";
    loadingState.hidden = name !== "loading";
    errorState.hidden = name !== "error";
    scoreView.hidden = name !== "score";
    uploadDetails.open = name !== "score";
    if (name !== "score") {
      sessionPanel.hidden = true;
    }
    scoreStage.setAttribute("aria-busy", name === "loading" ? "true" : "false");
    scoreStage.dataset.state = name;
  }

  function fileKind(file) {
    const name = file && file.name ? file.name : "";
    if (MIDI_NAME.test(name)) return "midi";
    if (AUDIO_NAME.test(name)) return "audio";
    if (file && typeof file.type === "string" && file.type.startsWith("audio/")) {
      return file.type === "audio/midi" || file.type === "audio/x-midi" ? "midi" : "audio";
    }
    return null;
  }

  function validateFile(kind, file) {
    const label = kind === "audio" ? "recording" : "MIDI transcription";
    const maxBytes = kind === "audio" ? MAX_AUDIO_BYTES : MAX_MIDI_BYTES;
    const maxMb = kind === "audio" ? MAX_AUDIO_MB : MAX_MIDI_MB;
    const pattern = kind === "audio" ? AUDIO_NAME : MIDI_NAME;
    if (!pattern.test(file.name || "")) {
      return kind === "audio"
        ? "Choose an audio recording (WAV, FLAC, OGG, MP3, M4A)."
        : "Choose a .mid or .midi file for the transcription.";
    }
    if (!file.size) {
      return "That file is empty.";
    }
    if (file.size > maxBytes) {
      return `That ${label} is larger than ${maxMb} MB.`;
    }
    return "";
  }

  function setSlot(kind, file) {
    const slot = slots[kind];
    const problem = validateFile(kind, file);
    if (problem) {
      setStatus(problem);
      return false;
    }

    slot.file = file;
    slot.row.classList.add("has-file");
    slot.title.textContent = file.name;
    slot.title.title = file.name;
    slot.meta.textContent = `${formatBytes(file.size)} · ready`;
    slot.action.textContent = "Replace";
    slot.clear.hidden = false;
    setStatus("");
    if (!errorState.hidden && !scoreReady) {
      setState("empty");
    }
    updateConvertAvailability();
    return true;
  }

  function resetSlot(kind) {
    const slot = slots[kind];
    slot.file = null;
    slot.input.value = "";
    slot.row.classList.remove("has-file", "is-dragging");
    slot.title.textContent = slot.defaultTitle;
    slot.title.removeAttribute("title");
    slot.meta.textContent = slot.defaultMeta;
    slot.action.textContent = slot.defaultAction;
    slot.clear.hidden = true;
  }

  function clearSlot(kind) {
    if (busy) return;
    resetSlot(kind);
    setStatus("");
    updateConvertAvailability();
    if (!scoreReady && errorState.hidden === false) {
      setState("empty");
    }
  }

  function updateConvertAvailability() {
    const pairReady = Boolean(slots.audio.file && slots.midi.file);
    convertButton.disabled = busy || !serverReady || !pairReady;
    convertLabel.textContent = busy
      ? "Making tab"
      : lastConverted && pairReady
        ? "Convert again"
        : "Make tab";
  }

  function setBusy(value) {
    busy = value;
    convertButton.classList.toggle("is-loading", value);
    audioInput.disabled = value;
    midiInput.disabled = value;
    audioClear.disabled = value;
    midiClear.disabled = value;
    cancelButton.disabled = !value;
    cancelButton.hidden = !value;
    form.setAttribute("aria-busy", value ? "true" : "false");
    audioRow.setAttribute("aria-disabled", value ? "true" : "false");
    midiRow.setAttribute("aria-disabled", value ? "true" : "false");
    if (value) {
      audioRow.classList.remove("is-dragging");
      midiRow.classList.remove("is-dragging");
      abortController = new AbortController();
    } else {
      abortController = null;
    }
    updateConvertAvailability();
    updatePlayAvailability();
  }

  function chooseFiles(files, fallbackKind = null) {
    if (busy) return;
    const list = Array.from(files || []);
    if (!list.length) return;

    let accepted = 0;
    for (const file of list) {
      const kind = fileKind(file) || (list.length === 1 ? fallbackKind : null);
      if (!kind) {
        setStatus(`“${file.name}” is not a MIDI or audio file.`);
        continue;
      }
      if (setSlot(kind, file)) accepted += 1;
    }
    if (accepted && slots.audio.file && slots.midi.file) {
      setStatus("");
    }
  }

  function setServerState(ok) {
    serverReady = Boolean(ok);
    serverDot.dataset.state = ok ? "ok" : "down";
    serverLabel.textContent = ok ? "local server" : "unavailable";
    updateConvertAvailability();
    if (!ok) {
      setStatus("The server isn’t responding. Is the Tabula workbench running?");
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
      // Some builds clamp or ignore; keep the control responsive.
    }
    speedSelect.value = String(playbackSpeed);
    try { localStorage.setItem("tabula:playbackSpeed", String(playbackSpeed)); } catch {}
  }

  function applyVolume() {
    if (!api) return;
    try {
      api.player.masterVolume = isMuted ? 0 : masterVolume;
    } catch (error) {
      // ignore a player that has not finished attaching yet
    }
    volumeSlider.disabled = !api.isReadyForPlayback || isMuted;
    volumeSlider.value = String(isMuted ? 0 : masterVolume);
    muteButton.setAttribute("aria-pressed", isMuted ? "true" : "false");
    muteButton.classList.toggle("is-muted", isMuted);
    try {
      localStorage.setItem("tabula:masterVolume.v2", String(masterVolume));
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
        enablePlayer: true,
        playerMode: PlayerMode ? PlayerMode.EnabledSynthesizer : 2,
        soundFont: "/static/alphatab/soundfont/sonivox.sf2",
        scrollElement: alphaTabEl,
        scrollMode: ScrollMode ? ScrollMode.Off : 0,
        enableCursor: true,
        enableAnimatedBeatCursor: true,
        enableElementHighlighting: false,
        enableUserInteraction: true,
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
      applyVolume();
    });

    if (api.soundFontLoadFailed && api.soundFontLoadFailed.on) {
      api.soundFontLoadFailed.on(() => {
        audioProblem = "The guitar sound could not load; the tab is still viewable.";
        updatePlayAvailability();
      });
    }

    api.playerStateChanged.on((args) => {
      setPlaying(Boolean(args && args.state === 1));
      updatePlayAvailability();
    });

    api.playerFinished.on(() => {
      setPlaying(false);
      if (loopEnabled && bars.length && currentBar >= 0) {
        const bar = bars[currentBar];
        const startTick = bar.start;
        try {
          api.tickPosition = startTick;
          api.play();
          setPlaying(true);
        } catch {}
        lastTick = startTick;
        currentTimeEl.textContent = formatTime(
          (startTick / Math.max(1, lastEndTick)) * lastEndTime
        );
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
          seek.value = String(
            Math.max(0, Math.min(1000, Math.round((tick / lastEndTick) * 1000)))
          );
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
      updatePlayAvailability();
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

  function finiteNumber(value) {
    if (value === null || value === undefined || value === "") return null;
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  // ---------- Recent takes (localStorage) ----------

  function getRecent() {
    try {
      const parsed = JSON.parse(localStorage.getItem("tabula:recent") || "[]");
      return Array.isArray(parsed) ? parsed : [];
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
    recentSection.hidden = arr.length === 0;
    recentList.hidden = arr.length === 0;
    recentList.innerHTML = arr
      .map((entry) => {
        const dead = Number(entry.dead_notes) || 0;
        const detail = [`${entry.notes || 0} notes`];
        if (dead > 0) detail.push(`${dead} dead`);
        return `<button type="button" class="recent-item" data-id="${escapeHtml(
          entry.id
        )}" title="${escapeHtml(entry.name)}" aria-label="Open ${escapeHtml(entry.name)}">
            <span class="recent-name">${escapeHtml(entry.name)}</span>
            <span class="recent-meta">${escapeHtml(detail.join(" · "))}</span>
          </button>`;
      })
      .join("");
    recentList.querySelectorAll(".recent-item").forEach((button) => {
      button.addEventListener("click", () => {
        const id = button.dataset.id;
        const entry = getRecent().find((item) => item.id === id);
        if (entry) restoreRecent(entry);
      });
    });
  }

  function addRecent(entry) {
    const arr = getRecent().filter((item) => item.id !== entry.id);
    arr.unshift(entry);
    setRecent(arr);
    renderRecent();
  }

  function restoreRecent(entry) {
    if (busy) return;
    if (!entry.urls || !entry.urls.gp5) {
      showError("That take is no longer available.", "The saved job has no tab file.");
      return;
    }
    const summary = {
      notes: Number(entry.notes) || 0,
      capo: Number(entry.capo) || 0,
      open_notes: Number(entry.open_notes) || 0,
      high_notes: Number(entry.high_notes) || 0,
      dropped: Number(entry.dropped) || 0,
      dead_notes: Number(entry.dead_notes) || 0,
      hp: Number(entry.hp) || 0,
      match_rate: entry.match_rate,
      offset_sec: entry.offset_sec,
      exported: { duration: Number(entry.duration) || 0, skipped: 0 },
    };
    const payload = {
      id: entry.id,
      title: entry.name,
      files: entry.files || {},
      urls: entry.urls,
      summary,
      elapsed: 0,
    };
    lastFileName = entry.name;
    lastConverted = true;
    populateSession(payload, entry.name);
    setState("score");
    loadScore(payload);
    updateConvertAvailability();
  }

  // ---------- End recent takes ----------

  function populateSession(payload, fileName) {
    const summary = payload.summary || {};
    const exported = summary.exported || {};
    const files = payload.files || {};
    const notes = Number(summary.notes) || 0;
    const openNotes = Number(summary.open_notes) || 0;
    const highNotes = Number(summary.high_notes) || 0;
    const deadNotes = Number(summary.dead_notes) || 0;
    const hpCount = Number(summary.hp) || 0;
    const dropped = Number(summary.dropped) || 0;
    const capo = Number(summary.capo) || 0;

    const title = payload.title || fileName || "Take";
    sessionFile.textContent = title;
    sessionFile.title = title;
    const sourceNames = [files.audio, files.midi].filter(Boolean);
    sessionSource.textContent = sourceNames.join("  +  ");
    sessionSource.title = sessionSource.textContent;
    sessionSource.hidden = sourceNames.length === 0;

    setFact(factCapo, capo > 0 ? `Fret ${capo}` : "None");
    setFact(factNotes, notes ? notes.toLocaleString() : "—");
    const matchRate = finiteNumber(summary.match_rate);
    const hasAlignment = matchRate !== null;
    setFact(
      factDead,
      hasAlignment ? deadNotes.toLocaleString() : "—",
      hasAlignment
        ? "Percussive dead notes inserted from the recording."
        : "Only available for audio + MIDI takes."
    );
    setFact(
      factHp,
      hasAlignment ? hpCount.toLocaleString() : "—",
      hasAlignment
        ? "Hammer-ons and pull-offs inferred between notes on the same string."
        : "Only available for audio + MIDI takes."
    );
    setFact(
      factOpen,
      notes ? `${openNotes.toLocaleString()} · ${percent(summary.open_rate, openNotes, notes)}` : "—",
      "Notes played on open strings after the capo."
    );
    const maxFret = finiteNumber(summary.max_fret);
    setFact(factMax, maxFret === null ? "—" : String(maxFret));
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
    if (hasAlignment) {
      const offset = Number(summary.offset_sec);
      const offsetText = Number.isFinite(offset) ? ` · offset ${offset >= 0 ? "+" : ""}${offset.toFixed(3)}s` : "";
      setFact(
        factMatch,
        `${(Math.max(0, Math.min(1, matchRate)) * 100).toFixed(1)}%`,
        `Audio onsets matched to MIDI notes${offsetText}.`
      );
    } else {
      setFact(factMatch, "—", "Only available for audio + MIDI takes.");
    }

    const duration = Number(exported.duration);
    lastEndTime = Number.isFinite(duration) && duration > 0 ? duration * 1000 : 0;
    totalTimeEl.textContent = formatTime(lastEndTime);
    currentTimeEl.textContent = "0:00";
    seek.value = "0";

    const slug = slugify(title || "tabula");
    downloadGp5.href = payload.urls.gp5;
    downloadGp5.setAttribute("download", `${slug}.gp5`);
    downloadMidi.href = payload.urls.midi;
    downloadMidi.setAttribute("download", `${slug}.mid`);
    downloadTab.href = payload.urls.tab;
    downloadTab.setAttribute("download", `${slug}_tab.txt`);
    downloadReport.href = payload.urls.report;
    downloadReport.setAttribute("download", `${slug}_report.json`);
    if (payload.urls.audio) {
      downloadAudio.href = payload.urls.audio;
      downloadAudio.setAttribute("download", `${slug}_recording.wav`);
      downloadAudio.hidden = false;
    } else {
      downloadAudio.hidden = true;
      downloadAudio.removeAttribute("href");
    }

    const notesOut = [];
    const skipped = Number(exported.skipped) || 0;
    if (skipped > 0) {
      notesOut.push(
        `${skipped} note${skipped === 1 ? "" : "s"} could not be placed in the Guitar Pro file; the text tab is complete.`
      );
    }
    if (deadNotes > 0) {
      notesOut.push(`${deadNotes} dead note${deadNotes === 1 ? "" : "s"} added.`);
    }
    if (hpCount > 0) {
      notesOut.push(
        `${hpCount} hammer-on/pull-off${hpCount === 1 ? "" : "s"} inferred.`
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

    currentBar = -1;
    lastTick = 0;
    lastEndTick = 0;
    const duration = Number((payload.summary && payload.summary.exported || {}).duration);
    lastEndTime = Number.isFinite(duration) && duration > 0 ? duration * 1000 : 0;
    audioProblem = "";
    barReadout.textContent = "Bar —";
    totalTimeEl.textContent = formatTime(lastEndTime);
    currentTimeEl.textContent = "0:00";
    seek.value = "0";
    updatePlayAvailability();
    scoreLoading.hidden = false;
    setPlayerStatus("Laying out the tab…");

    if (!api) {
      showError("The tab renderer is not ready.", "alphaTab has not initialized.");
      return;
    }

    let loaded = false;
    try {
      loaded = api.load(payload.urls.gp5);
    } catch (error) {
      showError("The tab could not be opened.", error && error.message ? error.message : String(error));
      return;
    }
    if (!loaded) {
      showError(
        "The tab could not be opened.",
        "alphaTab refused to load the generated Guitar Pro file."
      );
    }
  }

  function handleConverted(payload) {
    const fileName = payload.title || lastFileName || "";
    lastFileName = fileName;
    populateSession(payload, fileName);
    setState("score");
    loadScore(payload);
    lastConverted = true;
    updateConvertAvailability();

    const summary = payload.summary || {};
    const exported = summary.exported || {};
    addRecent({
      id: payload.id || slugify(fileName),
      name: fileName || "tabula",
      date: Date.now(),
      duration: Number(exported.duration) || 0,
      notes: Number(summary.notes) || 0,
      capo: Number(summary.capo) || 0,
      open_notes: Number(summary.open_notes) || 0,
      high_notes: Number(summary.high_notes) || 0,
      dead_notes: Number(summary.dead_notes) || 0,
      hp: Number(summary.hp) || 0,
      match_rate: summary.match_rate,
      offset_sec: summary.offset_sec,
      dropped: Number(summary.dropped) || 0,
      files: payload.files || {},
      urls: {
        gp5: payload.urls.gp5,
        midi: payload.urls.midi,
        tab: payload.urls.tab,
        report: payload.urls.report,
        audio: payload.urls.audio,
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
    if (busy || !serverReady) {
      return;
    }
    if (!slots.audio.file || !slots.midi.file) {
      setStatus("Add the recording and its MIDI transcription first.");
      return;
    }

    lastFileName = slots.audio.file.name.replace(/\.[^.]+$/, "") || slots.midi.file.name;
    setStatus("");
    if (api) {
      try {
        api.stop();
      } catch (error) {
        // A player that has not loaded a score yet can throw on stop.
      }
    }
    setPlaying(false);
    setState("loading");
    setBusy(true);

    const body = new FormData();
    body.append("audio", slots.audio.file, slots.audio.file.name);
    body.append("midi", slots.midi.file, slots.midi.file.name);

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
        setStatus("Conversion cancelled.", "info");
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
    resetSlot("audio");
    resetSlot("midi");
    setStatus("");
    lastConverted = false;
    lastFileName = "";
    scoreReady = false;
    audioProblem = "";
    score = null;
    bars = [];
    currentBar = -1;
    lastTick = 0;
    lastEndTick = 0;
    lastEndTime = 0;
    setState("empty");
    updateConvertAvailability();
  }

  function seekBySeconds(delta) {
    if (!api || !scoreReady || !lastEndTick || !lastEndTime) return;
    seeking = true;
    const now = (lastTick / Math.max(1, lastEndTick)) * lastEndTime;
    const targetMs = Math.max(0, Math.min(lastEndTime, now + delta * 1000));
    const section = Math.max(0, Math.min(1000, Math.round((targetMs / lastEndTime) * 1000)));
    seek.value = String(section);
    const tick = Math.round((section / 1000) * lastEndTick);
    currentTimeEl.textContent = formatTime(targetMs);
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

  async function loadJobFromUrl(jobParam, params) {
    const jobId = jobParam;
    try {
      const reportRes = await fetch(`/api/jobs/${jobId}/report.json`, {
        cache: "no-store",
        headers: { Accept: "application/json" },
      });
      if (!reportRes.ok) {
        throw new Error(`The job report returned HTTP ${reportRes.status}.`);
      }
      const report = await reportRes.json();
      const files = report.files || {};
      const summary = report.summary || {
        capo: Number(report.capo) || 0,
        notes: Number(report.final_notes) || 0,
        open_notes: 0,
        high_notes: 0,
        dropped: Array.isArray(report.dropped_chord_tones) ? report.dropped_chord_tones.length : 0,
        dead_notes: 0,
        hp: 0,
        exported: report.exported || {},
      };
      const title = files.title || report.title || `Take ${jobId.slice(0, 8)}`;
      const payload = {
        id: jobId,
        title,
        files,
        elapsed: 0,
        summary,
        urls: {
          gp5: `/api/jobs/${jobId}/score.gp5`,
          midi: `/api/jobs/${jobId}/final.mid`,
          tab: `/api/jobs/${jobId}/final_tab.txt`,
          report: `/api/jobs/${jobId}/report.json`,
        },
      };
      if (files.audio) {
        payload.urls.audio = `/api/jobs/${jobId}/source.wav`;
      }

      lastFileName = title;
      lastConverted = true;
      populateSession(payload, title);
      setState("score");
      loadScore(payload);
      updateConvertAvailability();

      const playParam = params.get("play");
      const seekParam = params.get("seek");
      if (playParam === "1") {
        const ready = window.setInterval(() => {
          if (api && scoreReady && api.isReadyForPlayback) {
            window.clearInterval(ready);
            if (seekParam) {
              const fraction = Math.max(0, Math.min(1, Number(seekParam)));
              seek.value = String(Math.round(fraction * 1000));
              const tick = Math.round(fraction * lastEndTick);
              currentTimeEl.textContent = formatTime(fraction * lastEndTime);
              updateMeasure(tick, true);
            }
            applyPlaybackSpeed();
            applyVolume();
            api.play();
            setPlaying(true);
          }
        }, 200);
      }
    } catch (error) {
      showError(
        "That take could not be loaded.",
        error && error.message ? error.message : String(error)
      );
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    convertFile();
  });

  audioInput.addEventListener("change", () => {
    chooseFiles(audioInput.files, "audio");
    audioInput.value = "";
  });

  midiInput.addEventListener("change", () => {
    chooseFiles(midiInput.files, "midi");
    midiInput.value = "";
  });

  audioClear.addEventListener("click", () => clearSlot("audio"));
  midiClear.addEventListener("click", () => clearSlot("midi"));

  Object.values(slots).forEach((slot) => {
    ["dragenter", "dragover"].forEach((type) => {
      slot.row.addEventListener(type, (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (!busy) {
          slot.row.classList.add("is-dragging");
        }
      });
    });
    ["dragleave", "dragend"].forEach((type) => {
      slot.row.addEventListener(type, (event) => {
        event.preventDefault();
        event.stopPropagation();
        slot.row.classList.remove("is-dragging");
      });
    });
    slot.row.addEventListener("drop", (event) => {
      event.preventDefault();
      event.stopPropagation();
      slot.row.classList.remove("is-dragging");
      if (busy) return;
      chooseFiles(event.dataTransfer && event.dataTransfer.files, slot.kind);
    });
  });

  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("drop", (event) => event.preventDefault());

  retryButton.addEventListener("click", () => {
    startOver();
    audioInput.click();
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
      return;
    }
    if (lastEndTick > 0 && lastTick >= lastEndTick - 1) {
      api.tickPosition = bars.length ? bars[0].start : 0;
    }
    applyPlaybackSpeed();
    applyVolume();
    try {
      api.play();
      setPlaying(true);
      setPlayerStatus("");
    } catch (error) {
      setPlayerStatus("Playback could not start.");
    }
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
    updatePlayAvailability();
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
    const active = document.activeElement;
    const nativeSpace = event.key === " "
      && active
      && (active.tagName === "BUTTON" || active.tagName === "A" || active.tagName === "SUMMARY");
    if (nativeSpace || isInteractiveInput(active)) return;
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

  setFollow(followEnabled);
  setLoop(loopEnabled);
  initAlphaTab();
  checkHealth();
  renderRecent();

  const params = new URLSearchParams(location.search);
  const jobParam = params.get("job");
  if (jobParam) {
    setState("loading");
    loadJobFromUrl(jobParam, params);
  } else {
    setState("empty");
  }
})();
