import * as AlphaTab from "/static/alphatab/alphaTab.min.mjs";

(() => {
  "use strict";

  const byId = (id) => document.getElementById(id);

  const form = byId("upload-form");
  const clientLocal = document.body.dataset.clientLocal !== "0";
  const audioInput = byId("audio-file");
  const audioRow = byId("audio-row");
  const audioTitle = byId("audio-title");
  const audioMeta = byId("audio-meta");
  const audioAction = byId("audio-action");
  const audioClear = byId("audio-clear");
  const audioUrlRow = byId("audio-url-row");
  const audioUrlInput = byId("audio-url");
  const audioUrlClear = byId("audio-url-clear");
  const convertButton = byId("convert-button");
  const convertLabel = byId("convert-label");
  const cancelButton = byId("cancel-button");
  const formStatus = byId("form-status");
  const capoSelect = byId("capo-select");
  const guitarSelect = byId("guitar-select");
  const beamSelect = byId("beam-select");
  const quantizeSelect = byId("quantize-select");
  const deadThresholdInput = byId("dead-threshold");

  const emptyState = byId("empty-state");
  const loadingState = byId("loading-state");
  const errorState = byId("error-state");
  const errorMessage = byId("error-message");
  const errorDetails = byId("error-details");
  const errorLog = byId("error-log");
  const retryButton = byId("retry-button");
  const streamPreview = byId("stream-preview");
  const streamMeta = byId("stream-meta");
  const streamStatus = byId("stream-status");
  const streamTab = byId("stream-tab");
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
  const staveButton = byId("stave-button");
  const speedSelect = byId("speed-select");
  const muteButton = byId("mute-button");
  const volumeSlider = byId("volume-slider");

  const sessionPanel = byId("session-panel");
  const takeTitleInput = byId("take-title");
  const titleStatus = byId("title-status");
  const sessionSource = byId("session-source");
  const workspace = document.querySelector(".workspace");
  const railResizer = byId("rail-resizer");
  const factCapo = byId("fact-capo");
  const factNotes = byId("fact-notes");
  const factDead = byId("fact-dead");
  const factHp = byId("fact-hp");
  const factOpen = byId("fact-open");
  const factMax = byId("fact-max");
  const factHigh = byId("fact-high");
  const factDropped = byId("fact-dropped");
  const factMatch = byId("fact-match");
  const downloadPdf = byId("download-pdf");
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
  const MAX_AUDIO_BYTES = MAX_AUDIO_MB * 1024 * 1024;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  const AUDIO_NAME = /\.(wav|flac|ogg|oga|mp3|m4a|aac|aif|aiff|opus|wma)$/i;

  // PDF export stays out of the initial page weight: the libraries are fetched
  // the first time the user asks for a PDF.
  const PDF_LIBS = [
    "/static/vendor/jspdf.umd.min.js",
    "/static/vendor/svg2pdf.umd.min.js",
  ];
  const A4_WIDTH = 595.28;
  const A4_HEIGHT = 841.89;
  const PDF_MARGIN = 34; // 12 mm
  const PDF_SYSTEM_GAP = 10;
  const JOB_POLL_MS = 700;

  const LayoutMode = AlphaTab && AlphaTab.LayoutMode;
  const ScrollMode = AlphaTab && AlphaTab.ScrollMode;
  const StaveProfile = AlphaTab && AlphaTab.StaveProfile;
  const NotationElement = AlphaTab && AlphaTab.NotationElement;
  const PlayerMode = AlphaTab
    ? AlphaTab.PlayerMode || (AlphaTab.synth && AlphaTab.synth.PlayerMode)
    : null;

  let busy = false;
  let serverReady = false;
  let abortController = null;
  let progressTimer = null;
  let currentJobId = "";
  let queueWait = false;
  let pdfBusy = false;
  let pdfLibsPromise = null;
  let provisionalLoaded = false;
  let lastConverted = false;
  let lastFileName = "";
  let currentTitle = "";
  let savingTitle = false;
  let titleStatusTimer = 0;

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
  let tabOnly = localStorage.getItem("tabula:tabOnly") !== "0";
  let programmaticScrollUntil = 0;

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

  function resetStream() {
    if (progressTimer !== null) {
      window.clearInterval(progressTimer);
      progressTimer = null;
    }
    queueWait = false;
    streamPreview.hidden = true;
    streamMeta.textContent = "";
    streamStatus.textContent = "Waiting for the transcription…";
    streamStatus.hidden = false;
    streamTab.textContent = "";
    streamTab.hidden = true;
    loadingState.classList.remove("has-stream");
  }

  function makeJobId() {
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      return window.crypto.randomUUID().replace(/-/g, "");
    }
    const seed = `${Date.now().toString(16)}${Math.random().toString(16).slice(2)}`;
    return `${seed}${"0".repeat(32)}`.slice(0, 32);
  }

  function updateStreamProgress(payload) {
    if (!payload) return;
    // The queue position would otherwise be overwritten by the older progress
    // file the server writes before the job starts.
    if (queueWait) return;
    const tab = typeof payload.tab === "string" ? payload.tab : "";
    const notes = Number(payload.notes) || 0;
    const done = Number(payload.chunks_done) || 0;
    const total = Number(payload.chunks_total) || 0;
    const capo = Number(payload.capo) || 0;
    const capoText = payload.auto_capo
      ? "auto capo preview"
      : capo > 0
        ? `Capo ${capo}`
        : "No capo";
    const chunkText = total ? `chunk ${done}/${total} · ` : "";
    const stageText = payload.message ? String(payload.message) : "";
    streamPreview.hidden = false;
    loadingState.classList.add("has-stream");
    if (tab.trim()) {
      streamStatus.hidden = true;
      streamTab.hidden = false;
      streamTab.textContent = tab;
      streamMeta.textContent = `${chunkText}${notes} notes · ${capoText}${payload.final ? " · final" : ""}`;
      streamTab.scrollTop = streamTab.scrollHeight;
      if (payload.provisional) showProvisionalScore(payload);
    } else {
      streamTab.hidden = true;
      streamTab.textContent = "";
      streamStatus.hidden = false;
      streamStatus.textContent = stageText || "Preparing the tab…";
      streamMeta.textContent = payload.stage === "transcribe" ? "audio transcription" : "preparing";
    }
  }

  function startStreamProgress(jobId) {
    currentJobId = jobId;
    provisionalLoaded = false;
    resetStream();
    const poll = async () => {
      if (!busy) return;
      try {
        const response = await fetch(`/api/jobs/${jobId}/progress.json`, {
          cache: "no-store",
          headers: { Accept: "application/json" },
        });
        if (!response.ok) return;
        updateStreamProgress(await response.json());
      } catch (error) {
        // Progress is best-effort; conversion errors still arrive on the main request.
      }
    };
    progressTimer = window.setInterval(poll, 400);
    poll();
  }

  function abortError() {
    const error = new Error("Cancelled");
    error.name = "AbortError";
    return error;
  }

  // Resolves after `ms`, but rejects as soon as the conversion is cancelled.
  function wait(ms, signal) {
    return new Promise((resolve, reject) => {
      if (signal && signal.aborted) {
        reject(abortError());
        return;
      }
      const onAbort = () => {
        window.clearTimeout(timer);
        reject(abortError());
      };
      const timer = window.setTimeout(() => {
        if (signal) signal.removeEventListener("abort", onAbort);
        resolve();
      }, ms);
      if (signal) signal.addEventListener("abort", onAbort, { once: true });
    });
  }

  function isConvertResult(payload) {
    return Boolean(payload && payload.urls && payload.urls.gp5);
  }

  function showQueueWait(position) {
    const place = Number(position);
    queueWait = true;
    streamPreview.hidden = false;
    loadingState.classList.add("has-stream");
    streamTab.hidden = true;
    streamTab.textContent = "";
    streamStatus.hidden = false;
    streamStatus.textContent =
      Number.isFinite(place) && place > 0
        ? `Waiting in queue (position ${place})…`
        : "Waiting in queue…";
    streamMeta.textContent = "queued";
  }

  function showStreamMessage(message) {
    if (!message || !streamTab.hidden) return;
    streamPreview.hidden = false;
    loadingState.classList.add("has-stream");
    streamStatus.hidden = false;
    streamStatus.textContent = message;
    streamMeta.textContent = "working";
  }

  // POST /api/convert now queues the job and returns 202; the result arrives
  // through status.json. Servers without that endpoint return it inline.
  async function awaitJob(jobId, initial, signal) {
    if (isConvertResult(initial)) return initial;
    if (!jobId) throw new Error("The converter returned an incomplete result.");

    currentJobId = jobId;
    startStreamProgress(jobId);

    let misses = 0;
    while (true) {
      let response = null;
      try {
        response = await fetch(`/api/jobs/${jobId}/status.json`, {
          cache: "no-store",
          headers: { Accept: "application/json" },
          signal,
        });
      } catch (error) {
        if (error && error.name === "AbortError") throw error;
        response = null;
      }

      if (response && response.status === 404) {
        // A job id from an older server has no status file; trust the POST.
        if (isConvertResult(initial)) return initial;
        throw new Error("The converter did not return a result.");
      }

      if (!response || !response.ok) {
        misses += 1;
        if (misses >= 5) {
          throw new Error(
            response
              ? `The converter status returned HTTP ${response.status}.`
              : "The converter stopped responding."
          );
        }
        await wait(JOB_POLL_MS, signal);
        continue;
      }

      misses = 0;
      let status = null;
      try {
        status = await response.json();
      } catch (error) {
        status = null;
      }
      if (!status) {
        await wait(JOB_POLL_MS, signal);
        continue;
      }

      if (status.state === "queued") {
        showQueueWait(status.queue_position);
      } else {
        queueWait = false;
        if (status.state === "running") showStreamMessage(status.message);
      }

      if (status.state === "done") {
        if (!isConvertResult(status.result)) {
          throw new Error("The converter returned an incomplete result.");
        }
        return status.result;
      }
      if (status.state === "error") {
        const failure = new Error(status.error || "The conversion failed.");
        failure.detail = status.detail || "";
        throw failure;
      }

      await wait(JOB_POLL_MS, signal);
    }
  }

  function setState(name) {
    emptyState.hidden = name !== "empty";
    loadingState.hidden = name !== "loading";
    errorState.hidden = name !== "error";
    scoreView.hidden = name !== "score";
    if (name !== "score") {
      sessionPanel.hidden = true;
    }
    if (name !== "loading") {
      resetStream();
    }
    scoreStage.setAttribute("aria-busy", name === "loading" ? "true" : "false");
    scoreStage.dataset.state = name;
  }

  function fileKind(file) {
    const name = file && file.name ? file.name : "";
    if (AUDIO_NAME.test(name)) return "audio";
    if (file && typeof file.type === "string" && file.type.startsWith("audio/")) {
      return "audio";
    }
    return null;
  }

  function validateFile(kind, file) {
    if (kind !== "audio" || !AUDIO_NAME.test(file.name || "")) {
      return "Choose an audio recording.";
    }
    if (!file.size) {
      return "That file is empty.";
    }
    if (file.size > MAX_AUDIO_BYTES) {
      return `That recording is larger than ${MAX_AUDIO_MB} MB.`;
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
    if (kind === "audio") {
      clearLink();
      syncLinkState();
    }
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

  function linkValue() {
    return audioUrlInput ? audioUrlInput.value.trim() : "";
  }

  function validLink(value) {
    if (!value) {
      return true;
    }
    try {
      const parsed = new URL(value);
      return parsed.protocol === "http:" || parsed.protocol === "https:";
    } catch (error) {
      return false;
    }
  }

  function clearLink() {
    if (!audioUrlInput) return;
    audioUrlInput.value = "";
    audioUrlClear.hidden = true;
    audioUrlRow.classList.remove("is-filled");
  }

  function syncLinkState() {
    if (!audioUrlInput) return;
    const value = linkValue();
    audioUrlClear.hidden = !value;
    audioUrlRow.classList.toggle("is-filled", Boolean(value));
    updateConvertAvailability();
  }

  function updateConvertAvailability() {
    const hasSource = Boolean(slots.audio.file) || Boolean(linkValue());
    convertButton.disabled = busy || !serverReady || !hasSource;
    convertLabel.textContent = busy
      ? linkValue() && !slots.audio.file
        ? "Downloading + making tab"
        : "Transcribing + making tab"
      : lastConverted && hasSource
        ? "Convert again"
        : "Transcribe + make tab";
  }

  function setBusy(value) {
    busy = value;
    convertButton.classList.toggle("is-loading", value);
    audioInput.disabled = value;
    audioClear.disabled = value;
    if (audioUrlInput) {
      audioUrlInput.disabled = value;
      audioUrlClear.disabled = value;
    }
    capoSelect.disabled = value;
    if (guitarSelect) guitarSelect.disabled = value;
    if (beamSelect) beamSelect.disabled = value;
    if (quantizeSelect) quantizeSelect.disabled = value;
    if (deadThresholdInput) deadThresholdInput.disabled = value;
    cancelButton.disabled = !value;
    cancelButton.hidden = !value;
    form.setAttribute("aria-busy", value ? "true" : "false");
    audioRow.setAttribute("aria-disabled", value ? "true" : "false");
    if (value) {
      audioRow.classList.remove("is-dragging");
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
        setStatus(`“${file.name}” is not an audio file.`);
        continue;
      }
      if (setSlot(kind, file)) accepted += 1;
    }
    if (accepted && slots.audio.file) {
      setStatus("");
    }
  }

  function setServerState(ok) {
    serverReady = Boolean(ok);
    serverDot.dataset.state = ok ? (clientLocal ? "ok" : "public") : "down";
    serverLabel.textContent = ok
      ? (clientLocal ? "local server" : "public demo")
      : "unavailable";
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
    updatePdfAvailability();
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

  // alphaTab always draws rest glyphs; the tab reads cleaner without them.
  // They are plain <text> nodes in the SMuFL rest range (U+E4E0..U+E4EF).
  function hideRestGlyphs() {
    if (!alphaTabEl) return;
    alphaTabEl.querySelectorAll("text").forEach((node) => {
      const code = node.textContent ? node.textContent.codePointAt(0) : undefined;
      if (
        code !== undefined &&
        code >= 0xe4e0 &&
        code <= 0xe4ef &&
        node.style.visibility !== "hidden"
      ) {
        node.style.visibility = "hidden";
      }
    });
  }

  let restObserver = null;

  function installRestHiding() {
    hideRestGlyphs();
    if (restObserver || !alphaTabEl || typeof MutationObserver !== "function") {
      return;
    }
    // Re-renders replace the SVG nodes; only watch childList so hiding a
    // glyph (an attribute change) cannot retrigger the observer.
    restObserver = new MutationObserver(() => hideRestGlyphs());
    restObserver.observe(alphaTabEl, { childList: true, subtree: true });
  }

  function setTabOnly(enabled) {
    tabOnly = Boolean(enabled);
    staveButton.setAttribute("aria-pressed", tabOnly ? "true" : "false");
    staveButton.classList.toggle("is-on", tabOnly);
    try {
      localStorage.setItem("tabula:tabOnly", tabOnly ? "1" : "0");
    } catch {}
    if (!api || !StaveProfile) return;
    try {
      api.settings.display.staveProfile = tabOnly
        ? StaveProfile.Tab
        : StaveProfile.Default;
      api.updateSettings();
      if (scoreReady) api.render();
    } catch (error) {
      // A player that is not ready yet picks the profile up on the next load.
    }
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
        staveProfile: tabOnly && StaveProfile ? StaveProfile.Tab : 0,
      },
      // The score is for reading the tab: no dynamic (f, p, mf) markings.
      notation: {
        elements: NotationElement
          ? new Map([[NotationElement.EffectDynamics, false]])
          : new Map(),
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
      installRestHiding();
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
    const viewport = alphaTabEl.clientHeight;
    const bottom = top + viewport;
    // Keep the active bar inside a generous safe zone.  A tiny margin made the
    // scrollbar fire on almost every bar and fight manual scrolling.
    const margin = Math.max(72, Math.min(180, viewport * 0.2));
    if (y < top + margin || y + bar.visualBounds.h > bottom - margin) {
      programmaticScrollUntil = performance.now() + 180;
      alphaTabEl.scrollTop = Math.max(0, y - margin);
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
        const d = entry.date ? new Date(entry.date) : null;
        const meta = d
          ? d.toLocaleDateString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })
          : "—";
        return `<button type="button" class="recent-item" data-id="${escapeHtml(
          entry.id
        )}" title="${escapeHtml(entry.name)}" aria-label="Open ${escapeHtml(entry.name)}">
            <span class="recent-name">${escapeHtml(entry.name)}</span>
            <span class="recent-meta">${escapeHtml(meta)}</span>
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

  function recentFromPayload(payload, id, name) {
    const summary = payload.summary || {};
    const exported = summary.exported || {};
    return {
      id: id,
      name: name || "tabula",
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
    };
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
    currentJobId = /^[0-9a-f]{32}$/.test(entry.id || "") ? entry.id : "";
    populateSession(payload, entry.name);
    setState("score");
    loadScore(payload);
    updateConvertAvailability();
  }

  // ---------- End recent takes ----------

  function updateDownloadNames(title) {
    const slug = slugify(title || "tabula");
    downloadPdf.setAttribute("download", `${slug}.pdf`);
    downloadMidi.setAttribute("download", `${slug}.mid`);
    downloadTab.setAttribute("download", `${slug}_tab.txt`);
    downloadReport.setAttribute("download", `${slug}_report.json`);
    if (!downloadAudio.hidden) {
      downloadAudio.setAttribute("download", `${slug}_recording.wav`);
    }
  }

  // ---------- PDF export ----------

  function updatePdfAvailability() {
    const ready = Boolean(scoreReady && !pdfBusy);
    downloadPdf.classList.toggle("is-disabled", !ready);
    downloadPdf.classList.toggle("is-busy", pdfBusy);
    downloadPdf.setAttribute("aria-disabled", ready ? "false" : "true");
  }

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = src;
      script.async = true;
      script.addEventListener("load", () => resolve());
      script.addEventListener("error", () => {
        script.remove();
        reject(new Error(`Could not load ${src}.`));
      });
      document.head.appendChild(script);
    });
  }

  function loadPdfLibs() {
    if (pdfLibsPromise) return pdfLibsPromise;
    pdfLibsPromise = PDF_LIBS.reduce(
      (chain, src) => chain.then(() => loadScript(src)),
      Promise.resolve()
    ).catch((error) => {
      // A failed load should be retryable on the next click.
      pdfLibsPromise = null;
      throw error;
    });
    return pdfLibsPromise;
  }

  function svgSize(svg) {
    const width = parseFloat(svg.getAttribute("width"));
    const height = parseFloat(svg.getAttribute("height"));
    if (Number.isFinite(width) && width > 0 && Number.isFinite(height) && height > 0) {
      return { width, height };
    }
    const box = svg.getBoundingClientRect();
    return { width: box.width, height: box.height };
  }

  function preparePdfSvg(svg, width, height) {
    const clone = svg.cloneNode(true);
    // alphaTab writes absolute pixel sizes but no viewBox; svg2pdf needs one
    // to scale the system into the page width.
    if (!clone.hasAttribute("viewBox")) {
      clone.setAttribute("viewBox", `0 0 ${width} ${height}`);
    }
    clone.removeAttribute("style");
    return clone;
  }

  async function exportPdf() {
    if (pdfBusy) return;
    if (!scoreReady) {
      setPlayerStatus("The tab is still loading; try the PDF again in a moment.");
      return;
    }

    const systems = Array.from(alphaTabEl.querySelectorAll("svg.at-surface-svg"));
    if (!systems.length) {
      setPlayerStatus("The score has not been laid out yet; try the PDF again in a moment.");
      return;
    }

    pdfBusy = true;
    updatePdfAvailability();
    setPlayerStatus("Preparing the PDF…");
    const title = currentTitle || lastFileName || "tabula";
    const fileName = `${slugify(title)}.pdf`;

    try {
      await loadPdfLibs();
      const JsPdf = window.jspdf && window.jspdf.jsPDF;
      // The UMD build exports a namespace object; accept either shape.
      const svgModule = window.svg2pdf;
      const renderSvg =
        typeof svgModule === "function" ? svgModule : svgModule && svgModule.svg2pdf;
      if (!JsPdf || typeof renderSvg !== "function") {
        throw new Error("The PDF libraries did not load.");
      }

      const doc = new JsPdf({
        orientation: "portrait",
        unit: "pt",
        format: "a4",
        compress: true,
      });

      const contentWidth = A4_WIDTH - PDF_MARGIN * 2;
      const contentBottom = A4_HEIGHT - PDF_MARGIN;
      let cursor = PDF_MARGIN;
      let drawn = 0;

      for (const svg of systems) {
        const { width, height } = svgSize(svg);
        if (!(width > 0) || !(height > 0)) continue;

        let drawWidth = contentWidth;
        let drawHeight = (height * contentWidth) / width;
        if (drawHeight > contentBottom - PDF_MARGIN) {
          drawHeight = contentBottom - PDF_MARGIN;
          drawWidth = (width * drawHeight) / height;
        }

        if (drawn > 0 && cursor + drawHeight > contentBottom) {
          doc.addPage();
          cursor = PDF_MARGIN;
        }

        await renderSvg(preparePdfSvg(svg, width, height), doc, {
          x: PDF_MARGIN,
          y: cursor,
          width: drawWidth,
          height: drawHeight,
        });
        cursor += drawHeight + PDF_SYSTEM_GAP;
        drawn += 1;
      }

      if (!drawn) {
        throw new Error("The score has no systems to export.");
      }

      doc.save(fileName);
      setPlayerStatus(`Saved ${fileName}`);
    } catch (error) {
      const reason = error && error.message ? error.message : "unknown error";
      setPlayerStatus(`The PDF could not be produced: ${reason}`);
    } finally {
      pdfBusy = false;
      updatePdfAvailability();
    }
  }

  function setTitleStatus(message) {
    window.clearTimeout(titleStatusTimer);
    titleStatus.textContent = message || "";
    if (message) {
      titleStatusTimer = window.setTimeout(() => {
        titleStatus.textContent = "";
      }, 2600);
    }
  }

  function updateRecentName(id, name) {
    const arr = getRecent();
    const entry = arr.find((item) => item.id === id);
    if (!entry) return;
    entry.name = name;
    setRecent(arr);
    renderRecent();
  }

  async function saveTakeTitle() {
    if (savingTitle) return;
    if (!currentJobId) {
      takeTitleInput.value = currentTitle;
      setTitleStatus("This take cannot be renamed.");
      return;
    }
    const title = takeTitleInput.value.trim().replace(/\s+/g, " ");
    if (!title) {
      takeTitleInput.value = currentTitle;
      return;
    }
    if (title === currentTitle) return;
    savingTitle = true;
    try {
      const response = await fetch(`/api/jobs/${currentJobId}/title`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ title }),
      });
      if (!response.ok) {
        throw new Error(`The title could not be saved (HTTP ${response.status}).`);
      }
      const data = await response.json();
      currentTitle = data.title || title;
      lastFileName = currentTitle;
      takeTitleInput.value = currentTitle;
      takeTitleInput.title = currentTitle;
      document.title = `${currentTitle} — Tabula`;
      updateDownloadNames(currentTitle);
      updateRecentName(currentJobId, currentTitle);
      if (score) {
        try {
          score.title = currentTitle;
          api.render();
        } catch (error) {
          // The file on disk is already updated; a reload will show it.
        }
      }
      setTitleStatus("Title saved.");
    } catch (error) {
      takeTitleInput.value = currentTitle;
      setTitleStatus(
        error && error.message ? error.message : "The title could not be saved."
      );
    } finally {
      savingTitle = false;
    }
  }

  function populateSession(payload, fileName) {
    const summary = payload.summary || {};
    const exported = summary.exported || {};
    const files = payload.files || {};
    const notes = Number(summary.notes) || 0;
    const openNotes = Number(summary.open_notes) || 0;
    const highNotes = Number(summary.high_notes) || 0;
    const deadNotes = Number(summary.dead_notes) || 0;
    const nonNoteCount = Number(summary.non_note) || 0;
    const hpCount = Number(summary.hp) || 0;
    const dropped = Number(summary.dropped) || 0;
    const capo = Number(summary.capo) || 0;

    const title = payload.title || fileName || "Take";
    currentTitle = title;
    takeTitleInput.value = title;
    takeTitleInput.title = title;
    document.title = `${title} — Tabula`;
    setTitleStatus("");
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
        ? "Dead notes inserted for percussion and other non-note hits."
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

    downloadMidi.href = payload.urls.midi;
    downloadTab.href = payload.urls.tab;
    downloadReport.href = payload.urls.report;
    if (payload.urls.audio) {
      downloadAudio.href = payload.urls.audio;
      downloadAudio.hidden = false;
    } else {
      downloadAudio.hidden = true;
      downloadAudio.removeAttribute("href");
    }
    updateDownloadNames(title);

    const notesOut = [];
    const skipped = Number(exported.skipped) || 0;
    const overlapTrimmed = Number(exported.overlap_trimmed) || 0;
    if (skipped > 0) {
      notesOut.push(
        `${skipped} note${skipped === 1 ? "" : "s"} could not be placed in the Guitar Pro file; the text tab is complete.`
      );
    }
    if (overlapTrimmed > 0) {
      notesOut.push(
        `${overlapTrimmed} same-string overlap${overlapTrimmed === 1 ? "" : "s"} trimmed for playback.`
      );
    }
    if (nonNoteCount > 0) {
      notesOut.push(
        `${nonNoteCount} matched non-note hit${nonNoteCount === 1 ? "" : "s"} converted to dead notes.`
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

  function showProvisionalScore(progress) {
    if (provisionalLoaded || !currentJobId) return;
    provisionalLoaded = true;
    const title = lastFileName || "Draft tab";
    const draft = {
      id: currentJobId,
      title,
      files: { title },
      urls: {
        gp5: `/api/jobs/${currentJobId}/provisional.gp5`,
        midi: "",
        tab: "",
        report: "",
      },
      summary: { exported: { duration: Number(progress.duration) || 0 } },
      elapsed: 0,
    };
    setState("score");
    loadScore(draft);
    setPlayerStatus("Draft tab — adding dead notes…");
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

    addRecent(recentFromPayload(
      payload,
      payload.id || slugify(fileName),
      fileName || "tabula"
    ));

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
    const file = slots.audio.file;
    const link = linkValue();
    if (!file && !link) {
      setStatus("Add a recording or paste a link first.");
      return;
    }
    if (!file && !validLink(link)) {
      setStatus("Paste a full http:// or https:// link.");
      return;
    }

    lastFileName = file
      ? file.name.replace(/\.[^.]+$/, "") || "take"
      : "Linked take";
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
    resetStream();

    const jobId = makeJobId();
    const body = new FormData();
    if (file) {
      body.append("audio", file, file.name);
    } else {
      body.append("url", link);
    }
    body.append("capo", capoSelect.value);
    body.append("guitar", guitarSelect ? guitarSelect.value : "acoustic");
    body.append("beam", beamSelect ? beamSelect.value : "1");
    if (quantizeSelect) body.append("quantize", quantizeSelect.value);
    if (deadThresholdInput && deadThresholdInput.value !== "") {
      body.append("dead_threshold", deadThresholdInput.value);
    }
    body.append("job_id", jobId);

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

      if (response.status === 429) {
        const retryAfter = Math.max(
          1,
          Math.round(Number(response.headers.get("Retry-After")) || 0) || 30
        );
        setStatus(`The converter is busy, try again in ${retryAfter}s`);
        setState("empty");
        return;
      }

      if (!response.ok) {
        const message = (data && data.error) || `The converter returned HTTP ${response.status}.`;
        const failure = new Error(message);
        failure.detail = (data && data.detail) || "";
        throw failure;
      }

      if (!data || (!data.id && !isConvertResult(data))) {
        throw new Error("The converter returned an incomplete result.");
      }

      const result = await awaitJob(data.id, data, signal);
      if (abortController && abortController.signal.aborted) {
        return;
      }
      handleConverted(result);
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
      resetStream();
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
    clearLink();
    syncLinkState();
    setStatus("");
    lastConverted = false;
    lastFileName = "";
    currentJobId = "";
    currentTitle = "";
    takeTitleInput.value = "";
    document.title = "Tabula — audio to tab";
    setTitleStatus("");
    provisionalLoaded = false;
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
        non_note: 0,
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
      currentJobId = jobId;
      populateSession(payload, title);
      setState("score");
      loadScore(payload);
      updateConvertAvailability();
      addRecent(recentFromPayload(payload, jobId, title));

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

  audioClear.addEventListener("click", () => clearSlot("audio"));

  if (audioUrlInput) {
    audioUrlInput.addEventListener("input", () => {
      if (busy) return;
      if (linkValue() && slots.audio.file) {
        resetSlot("audio");
      }
      syncLinkState();
      if (!linkValue()) {
        setStatus("");
      } else if (!validLink(linkValue())) {
        setStatus("Paste a full http:// or https:// link.", "info");
      } else {
        setStatus("");
      }
      if (!scoreReady && !errorState.hidden) {
        setState("empty");
      }
    });
    audioUrlInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        convertFile();
      }
    });
    audioUrlClear.addEventListener("click", () => {
      clearLink();
      syncLinkState();
      setStatus("");
      if (!scoreReady && !errorState.hidden) {
        setState("empty");
      }
    });
  }

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
    const previousLink = linkValue();
    startOver();
    if (previousLink && audioUrlInput) {
      audioUrlInput.value = previousLink;
      syncLinkState();
    } else {
      audioInput.click();
    }
  });

  startOverButton.addEventListener("click", startOver);
  cancelButton.addEventListener("click", cancelConversion);

  downloadPdf.addEventListener("click", (event) => {
    event.preventDefault();
    exportPdf();
  });

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
  staveButton.addEventListener("click", () => setTabOnly(!tabOnly));

  // Manual scrolling takes priority over follow mode.  Without this, the next
  // bar boundary yanks the score back and the scrollbar appears to flicker.
  alphaTabEl.addEventListener(
    "scroll",
    () => {
      if (performance.now() < programmaticScrollUntil) return;
      if (followEnabled) setFollow(false);
    },
    { passive: true },
  );

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

  const isSpace = (event) => event.key === " " || event.code === "Space";

  window.addEventListener("keydown", (event) => {
    const active = document.activeElement;
    // Space always means play/pause, even if a toolbar button still has
    // focus from the last click.  Typing in a field still gets its space.
    if (isSpace(event) && !isInteractiveInput(active)) {
      event.preventDefault();
      playButton.click();
      return;
    }
    if (isInteractiveInput(active)) return;
    const key = event.key.toLowerCase();
    switch (key) {
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

  // Stop the browser from re-activating a focused button on key release,
  // which would undo the play/pause toggle above.
  window.addEventListener("keyup", (event) => {
    if (isSpace(event) && !isInteractiveInput(document.activeElement)) {
      event.preventDefault();
    }
  });

  takeTitleInput.addEventListener("change", saveTakeTitle);
  takeTitleInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      takeTitleInput.blur();
    } else if (event.key === "Escape") {
      takeTitleInput.value = currentTitle;
      takeTitleInput.blur();
    }
  });

  // ---------- Resizable session rail ----------

  const RAIL_DEFAULT = 360;
  const RAIL_MIN = 260;
  const RAIL_MAX = 560;
  let railDrag = null;
  let railRenderTimer = 0;

  function railMaxWidth() {
    return Math.max(RAIL_MIN, Math.min(RAIL_MAX, window.innerWidth - 420));
  }

  function currentRailWidth() {
    const value = parseFloat(workspace.style.getPropertyValue("--rail-width"));
    if (Number.isFinite(value) && value > 0) return value;
    const rail = workspace.querySelector(".control-rail");
    return rail ? rail.getBoundingClientRect().width : RAIL_DEFAULT;
  }

  function setRailWidth(width, persist) {
    const clamped = Math.round(
      Math.min(railMaxWidth(), Math.max(RAIL_MIN, width))
    );
    workspace.style.setProperty("--rail-width", `${clamped}px`);
    railResizer.setAttribute("aria-valuenow", String(clamped));
    railResizer.setAttribute("aria-valuemax", String(Math.round(railMaxWidth())));
    if (persist) {
      try {
        localStorage.setItem("tabula:railWidth", String(clamped));
      } catch {
        // storage unavailable; the width still applies for this session
      }
    }
  }

  function scheduleRailRender() {
    window.clearTimeout(railRenderTimer);
    railRenderTimer = window.setTimeout(() => {
      if (!api || !scoreReady) return;
      try {
        api.render();
      } catch (error) {
        // A render already in flight is harmless; leave the current layout.
      }
    }, 140);
  }

  const storedRail = Number(localStorage.getItem("tabula:railWidth"));
  if (Number.isFinite(storedRail) && storedRail > 0) {
    setRailWidth(storedRail, false);
  } else {
    setRailWidth(RAIL_DEFAULT, false);
  }

  railResizer.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    railDrag = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startWidth: currentRailWidth(),
    };
    try {
      railResizer.setPointerCapture(event.pointerId);
    } catch {}
    railResizer.classList.add("is-dragging");
    document.body.classList.add("is-resizing-rail");
    event.preventDefault();
  });

  railResizer.addEventListener("pointermove", (event) => {
    if (!railDrag || event.pointerId !== railDrag.pointerId) return;
    setRailWidth(railDrag.startWidth + event.clientX - railDrag.startX, false);
  });

  function endRailDrag(event) {
    if (!railDrag) return;
    if (event && event.pointerId !== railDrag.pointerId) return;
    try {
      railResizer.releasePointerCapture(railDrag.pointerId);
    } catch {}
    railDrag = null;
    railResizer.classList.remove("is-dragging");
    document.body.classList.remove("is-resizing-rail");
    setRailWidth(currentRailWidth(), true);
    scheduleRailRender();
  }

  railResizer.addEventListener("pointerup", endRailDrag);
  railResizer.addEventListener("pointercancel", endRailDrag);
  railResizer.addEventListener("dblclick", () => {
    setRailWidth(RAIL_DEFAULT, true);
    scheduleRailRender();
  });

  railResizer.addEventListener("keydown", (event) => {
    const step = event.shiftKey ? 48 : 16;
    const current = currentRailWidth();
    let next = null;
    if (event.key === "ArrowLeft") next = current - step;
    else if (event.key === "ArrowRight") next = current + step;
    else if (event.key === "Home") next = RAIL_MIN;
    else if (event.key === "End") next = railMaxWidth();
    if (next === null) return;
    event.preventDefault();
    setRailWidth(next, true);
    scheduleRailRender();
  });

  window.addEventListener("resize", () => {
    if (railDrag) return;
    if (currentRailWidth() > railMaxWidth()) {
      setRailWidth(railMaxWidth(), true);
    }
    scheduleRailRender();
  });

  setFollow(followEnabled);
  setTabOnly(tabOnly);
  setLoop(loopEnabled);
  updatePdfAvailability();
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
