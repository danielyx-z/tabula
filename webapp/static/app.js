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
    serverLabel.textContent = ok ? "local server" : "model files missing";
    if (!busy) {
      convertButton.disabled = !currentFile || !serverReady;
    }
    if (!ok) {
      setStatus("The model checkpoint or vocabulary is missing on this PC.");
    }
  }

  function setBusy(value) {
    busy = value;
    convertButton.classList.toggle("is-loading", value);
    convertLabel.textContent = value ? "Making tab" : "Make tab";
    convertButton.disabled = value || !currentFile || !serverReady;
    midiInput.disabled = value;
    clearFileButton.disabled = value;
    form.setAttribute("aria-busy", value ? "true" : "false");
    dropZone.setAttribute("aria-disabled", value ? "true" : "false");
    if (value) {
      dropZone.classList.remove("is-dragging");
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

    if (!errorState.hidden) {
      setState("empty");
    }
  }

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
      if (!scoreReady) {
        setPlayerStatus("");
      }
      return;
    }

    if (audioProblem) {
      playButton.disabled = true;
      seek.disabled = true;
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

    if (playable) {
      setPlayerStatus("");
    } else if (playerReady) {
      setPlayerStatus("Preparing audio…");
    } else {
      setPlayerStatus("Loading the player…");
    }
  }

  function setPlaying(playing) {
    isPlaying = Boolean(playing);
    // SVG elements do not reflect the `hidden` IDL property; toggle the
    // attribute so the play/pause icons actually swap.
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
        // Scroll the internal score viewport, not the page. alphaTab styles the
        // bar cursor; the app only adds a light tint and outline in CSS.
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
      updateMeasure(tick, false, true);
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
    if (reveal) {
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
    lastTick = 0;
    lastEndTick = 0;
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
    const fileName = lastFileName || (currentFile && currentFile.name) || "";
    populateSession(payload, fileName);
    setState("score");
    loadScore(payload);

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

    try {
      const response = await fetch("/api/convert", {
        method: "POST",
        body,
        headers: { Accept: "application/json" },
      });

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
      showError(
        (error && error.message) || "The conversion failed.",
        (error && error.detail) || ""
      );
    } finally {
      setBusy(false);
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
        setServerState(Boolean(data.ok && data.checkpoint_ready && data.vocab_ready));
      })
      .catch(() => {
        setServerState(false);
      });
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
    api.play();
    setPlaying(true);
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

  setState("empty");
  initAlphaTab();
  checkHealth();
})();
