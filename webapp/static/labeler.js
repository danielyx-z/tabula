(() => {
  "use strict";

  const KEY_LABEL = {
    "1": "pitched",
    "2": "dead_string",
    "3": "bass_flick",
    "4": "body_percussion",
  };

  const PLAY_BEFORE_S = 0.03;
  const PLAY_AFTER_S = 0.30;
  const PLAY_GUARD_S = 0.005;

  const els = {
    setup: document.getElementById("label-setup"),
    session: document.getElementById("label-session"),
    form: document.getElementById("label-upload-form"),
    file: document.getElementById("label-audio-file"),
    fileTitle: document.getElementById("label-file-title"),
    upload: document.getElementById("label-upload-button"),
    uploadLabel: document.getElementById("label-upload-label"),
    status: document.getElementById("label-status"),
    sessionStatus: document.getElementById("label-session-status"),
    name: document.getElementById("label-session-name"),
    progress: document.getElementById("label-progress-text"),
    time: document.getElementById("label-time-text"),
    bar: document.getElementById("label-progress-bar"),
    nowTime: document.getElementById("label-now-time"),
    nowNote: document.getElementById("label-now-note"),
    audio: document.getElementById("label-audio"),
    download: document.getElementById("label-download"),
    replay: document.getElementById("label-replay"),
    skip: document.getElementById("label-skip"),
    pitchedRun: document.getElementById("label-pitched-run"),
    undo: document.getElementById("label-undo"),
    keys: Array.from(document.querySelectorAll(".labeler-key")),
    recent: document.getElementById("label-recent"),
    recentList: document.getElementById("label-recent-list"),
  };

  const state = {
    session: null,
    labels: {},
    index: 0,
    busy: false,
    playing: false,
    stopAt: 0,
    raf: 0,
  };

  function formatTime(value) {
    const seconds = Math.max(0, Number(value) || 0);
    const minutes = Math.floor(seconds / 60);
    const rest = seconds - minutes * 60;
    return `${minutes}:${rest.toFixed(2).padStart(5, "0")}`;
  }

  function setStatus(text, isError = false) {
    els.status.textContent = text;
    els.status.classList.toggle("is-error", Boolean(isError));
  }

  function setSessionStatus(text, isError = false) {
    els.sessionStatus.textContent = text;
    els.sessionStatus.classList.toggle("is-error", Boolean(isError));
  }

  async function api(path, options = {}) {
    const response = await fetch(path, {
      headers: options.body && !(options.body instanceof FormData)
        ? { "Content-Type": "application/json" }
        : undefined,
      ...options,
    });
    let payload = null;
    try {
      payload = await response.json();
    } catch (_error) {
      payload = {};
    }
    if (!response.ok) {
      throw new Error(payload.error || payload.detail || "Request failed.");
    }
    return payload;
  }

  function labelMap(records) {
    const out = {};
    for (const record of records || []) {
      if (record && Number.isInteger(Number(record.index))) {
        out[Number(record.index)] = record;
      }
    }
    return out;
  }

  function nextUnlabeled(start) {
    if (!state.session) return -1;
    const total = state.session.candidates.length;
    for (let offset = 0; offset < total; offset += 1) {
      const index = (start + offset) % total;
      if (!(index in state.labels)) return index;
    }
    return -1;
  }

  function currentCandidate() {
    if (!state.session || state.index < 0) return null;
    return state.session.candidates[state.index] || null;
  }

  function stopPlayback() {
    state.playing = false;
    if (state.raf) cancelAnimationFrame(state.raf);
    state.raf = 0;
    try {
      els.audio.pause();
    } catch (_error) {
      // The browser can reject pause() only if the element is not ready.
    }
  }

  function tickPlayback() {
    if (!state.playing) return;
    if (els.audio.currentTime >= state.stopAt - 0.005 || els.audio.ended) {
      stopPlayback();
      return;
    }
    state.raf = requestAnimationFrame(tickPlayback);
  }

  async function playCurrent() {
    const candidate = currentCandidate();
    if (!candidate) return;
    stopPlayback();
    try {
      const onset = Number(candidate.onset_s) || 0;
      const previous = state.session.candidates[state.index - 1] || null;
      const next = state.session.candidates[state.index + 1] || null;
      const duration = Number(state.session.duration)
        || Number(els.audio.duration)
        || 0;
      let stopAt = onset + PLAY_AFTER_S;
      if (next) {
        stopAt = Math.min(stopAt, Number(next.onset_s) - PLAY_GUARD_S);
      }
      if (duration) stopAt = Math.min(stopAt, duration);
      let startAt = Math.max(0, onset - PLAY_BEFORE_S);
      if (previous) {
        const afterPrevious = Number(previous.onset_s) + PLAY_GUARD_S;
        if (afterPrevious < onset) startAt = Math.max(startAt, afterPrevious);
      }
      els.audio.currentTime = startAt;
      state.stopAt = Math.max(onset + 0.01, stopAt);
      state.playing = true;
      await els.audio.play();
      state.raf = requestAnimationFrame(tickPlayback);
    } catch (_error) {
      state.playing = false;
      setSessionStatus("Press space or Replay to hear the onset.");
    }
  }

  function render() {
    if (!state.session) return;
    const total = state.session.candidates.length;
    const labeled = Object.keys(state.labels).length;
    const candidate = currentCandidate();
    els.name.textContent = state.session.meta.original_name || state.session.session_id;
    els.progress.textContent = `${labeled} / ${total} labeled`;
    els.bar.style.width = total ? `${Math.min(100, (labeled / total) * 100)}%` : "0%";
    els.time.textContent = state.session.duration
      ? `${formatTime(state.session.duration)} total`
      : "—";
    els.download.href = `/api/label/sessions/${state.session.session_id}/labels.jsonl`;
    if (!candidate) {
      els.nowTime.textContent = "complete";
      els.nowNote.textContent = "All onsets have a label. Export the JSONL when ready.";
      els.sessionStatus.textContent = "Session complete.";
      return;
    }
    els.nowTime.textContent = formatTime(candidate.onset_s);
    const existing = state.labels[state.index];
    els.nowNote.textContent = existing
      ? `Already labeled: ${existing.label.replaceAll("_", " ")}`
      : "Press 1–4, or Shift+1 for a pitched run.";
  }

  function renderRecent(rows) {
    els.recentList.replaceChildren();
    if (!rows || !rows.length) {
      els.recent.hidden = true;
      return;
    }
    els.recent.hidden = false;
    for (const row of rows) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "labeler-recent-item";
      const name = document.createElement("span");
      name.className = "labeler-recent-name";
      name.textContent = row.original_name || row.session_id;
      const progress = document.createElement("span");
      progress.className = "labeler-recent-progress";
      progress.textContent = `${row.labeled || 0} / ${row.total || 0}`;
      button.append(name, progress);
      button.addEventListener("click", () => resumeSession(row.session_id));
      els.recentList.append(button);
    }
  }

  async function loadRecent() {
    try {
      const payload = await api("/api/label/sessions");
      renderRecent(payload.sessions || []);
    } catch (_error) {
      renderRecent([]);
    }
  }

  async function resumeSession(sessionId) {
    setStatus("Loading session…");
    try {
      const payload = await api(`/api/label/sessions/${sessionId}`);
      setStatus("");
      applySession(payload);
    } catch (error) {
      setStatus(error.message, true);
    }
  }

  function applySession(payload) {
    state.session = payload;
    state.labels = labelMap(payload.labels);
    state.index = nextUnlabeled(0);
    els.setup.hidden = true;
    els.session.hidden = false;
    els.audio.src = payload.audio_url;
    els.audio.load();
    render();
    if (state.index >= 0) playCurrent();
  }

  function moveTo(start) {
    state.index = nextUnlabeled(start);
    render();
    if (state.index >= 0) playCurrent();
  }

  function advance() {
    moveTo(state.index + 1);
  }

  async function saveLabel(label) {
    if (state.busy || !state.session || state.index < 0) return;
    state.busy = true;
    setSessionStatus("");
    try {
      const payload = await api(
        `/api/label/sessions/${state.session.session_id}/labels`,
        {
          method: "POST",
          body: JSON.stringify({ index: state.index, label }),
        },
      );
      state.session = payload;
      state.labels = labelMap(payload.labels);
      advance();
    } catch (error) {
      setSessionStatus(error.message, true);
    } finally {
      state.busy = false;
    }
  }

  async function savePitchedRun() {
    if (state.busy || !state.session || state.index < 0) return;
    state.busy = true;
    setSessionStatus("");
    try {
      const payload = await api(
        `/api/label/sessions/${state.session.session_id}/pitched-run`,
        {
          method: "POST",
          body: JSON.stringify({
            start_index: state.index,
            max_count: 8,
            max_gap_s: 0.25,
          }),
        },
      );
      state.session = payload;
      state.labels = labelMap(payload.labels);
      const indices = Array.isArray(payload.indices) ? payload.indices : [];
      const lastIndex = indices.length ? Number(indices[indices.length - 1]) : state.index;
      setSessionStatus(`Labeled ${indices.length || 1} normal attack(s) as pitched.`);
      moveTo(lastIndex + 1);
    } catch (error) {
      setSessionStatus(error.message, true);
    } finally {
      state.busy = false;
    }
  }

  async function undo() {
    if (state.busy || !state.session) return;
    state.busy = true;
    try {
      const payload = await api(
        `/api/label/sessions/${state.session.session_id}/undo`,
        { method: "POST", body: JSON.stringify({}) },
      );
      state.session = payload;
      state.labels = labelMap(payload.labels);
      const removed = payload.removed || null;
      const indices = removed && Array.isArray(removed.indices) ? removed.indices : [];
      const removedIndex = indices.length
        ? Number(indices[0])
        : (removed ? Number(removed.index) : -1);
      state.index = removedIndex >= 0 ? removedIndex : nextUnlabeled(0);
      render();
      if (state.index >= 0) playCurrent();
    } catch (error) {
      setSessionStatus(error.message, true);
    } finally {
      state.busy = false;
    }
  }

  els.form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = els.file.files && els.file.files[0];
    if (!file) return;
    els.upload.disabled = true;
    els.uploadLabel.textContent = "Splitting…";
    setStatus("Preparing audio and finding onsets…");
    const body = new FormData();
    body.append("audio", file);
    try {
      const payload = await api("/api/label/sessions", { method: "POST", body });
      setStatus("");
      applySession(payload);
    } catch (error) {
      setStatus(error.message, true);
    } finally {
      els.upload.disabled = false;
      els.uploadLabel.textContent = "Split and label";
    }
  });

  els.file.addEventListener("change", () => {
    const file = els.file.files && els.file.files[0];
    if (!file) return;
    els.fileTitle.textContent = file.name;
    els.upload.disabled = false;
  });

  for (const button of els.keys) {
    button.addEventListener("click", () => saveLabel(button.dataset.label));
  }
  els.replay.addEventListener("click", playCurrent);
  els.skip.addEventListener("click", () => {
    if (!state.session || state.index < 0) return;
    moveTo(state.index + 1);
  });
  els.pitchedRun.addEventListener("click", savePitchedRun);
  els.undo.addEventListener("click", undo);

  document.addEventListener("keydown", (event) => {
    if (event.repeat || !state.session) return;
    const tag = (event.target && event.target.tagName) || "";
    if (["INPUT", "TEXTAREA", "SELECT"].includes(tag)) return;
    if (event.shiftKey && (event.code === "Digit1" || event.key === "!")) {
      event.preventDefault();
      savePitchedRun();
      return;
    }
    if (event.key in KEY_LABEL) {
      event.preventDefault();
      saveLabel(KEY_LABEL[event.key]);
      return;
    }
    if (event.code === "Space") {
      event.preventDefault();
      playCurrent();
      return;
    }
    if (event.key.toLowerCase() === "s") {
      event.preventDefault();
      state.index = nextUnlabeled(state.index + 1);
      render();
      if (state.index >= 0) playCurrent();
      return;
    }
    if (event.key.toLowerCase() === "u") {
      event.preventDefault();
      undo();
    }
  });

  els.audio.addEventListener("ended", stopPlayback);

  const requestedSession = new URLSearchParams(window.location.search).get("session");
  loadRecent();
  if (requestedSession) {
    resumeSession(requestedSession);
  }
})();
