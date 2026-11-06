(() => {
  "use strict";

  const CONFIG = window.TABULA_LABEL || {};
  const form = document.getElementById("label-form");
  const audioInput = document.getElementById("label-audio");
  const midiInput = document.getElementById("label-midi");
  const audioTitle = document.getElementById("label-audio-title");
  const audioMeta = document.getElementById("label-audio-meta");
  const midiTitle = document.getElementById("label-midi-title");
  const midiMeta = document.getElementById("label-midi-meta");
  const startButton = document.getElementById("label-start");
  const status = document.getElementById("label-status");
  const upload = document.getElementById("label-upload");
  const workspace = document.getElementById("label-workspace");
  const canvas = document.getElementById("label-canvas");
  const wrap = document.getElementById("label-canvas-wrap");
  const empty = document.getElementById("label-empty");
  const playButton = document.getElementById("label-play");
  const timeLabel = document.getElementById("label-time");
  const zoomSlider = document.getElementById("label-zoom");
  const zoomInButton = document.getElementById("label-zoom-in");
  const zoomOutButton = document.getElementById("label-zoom-out");
  const scrollbar = document.getElementById("label-scrollbar");
  const scrollThumb = document.getElementById("label-scroll-thumb");
  const fitButton = document.getElementById("label-fit");
  const undoButton = document.getElementById("label-undo");
  const clearButton = document.getElementById("label-clear");
  const exportLink = document.getElementById("label-export");
  const reviewButton = document.getElementById("label-review");
  const newButton = document.getElementById("label-new");
  const deleteButton = document.getElementById("label-delete");
  const recent = document.getElementById("label-recent");
  const recentList = document.getElementById("label-recent-list");
  const nameLabel = document.getElementById("label-name");
  const countsLabel = document.getElementById("label-counts");
  const savedLabel = document.getElementById("label-saved");

  // Playback uses the Web Audio clock, not an <audio> element.  The media
  // element clock and its output latency are what made the old playhead drift
  // by a few tens of milliseconds; MuScriptor's own web player uses the audio
  // context clock for exactly this reason.
  let audioCtx = null;
  let audioBuffer = null;
  let sourceNode = null;
  let sourceStartCtx = 0;
  let sourceOffset = 0;
  let playing = false;
  let playhead = 0;
  let audioReady = false;
  let auditionEnd = null;

  const ctx = canvas.getContext("2d");
  const PAD = { left: 44, right: 14, top: 18, bottom: 28 };
  let session = null;
  let notes = [];
  let groups = [];
  let noteById = new Map();
  let view = { start: 0, end: 1 };
  let pitchMin = 40;
  let pitchMax = 80;
  let dragging = null;
  let selection = null;
  let pending = null;
  let undoStack = [];
  let dirty = false;
  let playFrame = 0;
  let hoverNote = null;

  function setStatus(message, kind = "error") {
    status.textContent = message || "";
    if (message) status.dataset.kind = kind;
    else delete status.dataset.kind;
  }

  function setSaved(value) {
    dirty = Boolean(value);
    savedLabel.textContent = dirty ? "Unsaved changes" : "All changes saved";
    savedLabel.classList.toggle("is-dirty", dirty);
  }

  function formatTime(seconds) {
    const value = Math.max(0, Number(seconds) || 0);
    return `${Math.floor(value / 60)}:${(value % 60).toFixed(1).padStart(4, "0")}`;
  }

  function ensureAudioContext() {
    if (!audioCtx) {
      const Ctor = window.AudioContext || window.webkitAudioContext;
      audioCtx = new Ctor({ latencyHint: "interactive" });
    }
    return audioCtx;
  }

  async function loadAudio(url) {
    const ctx = ensureAudioContext();
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Audio HTTP ${response.status}`);
    const data = await response.arrayBuffer();
    audioBuffer = await ctx.decodeAudioData(data);
    audioReady = true;
    playhead = 0;
    playButton.disabled = false;
  }

  function currentTime() {
    if (playing && audioCtx && audioBuffer) {
      return Math.min(
        audioBuffer.duration,
        sourceOffset + Math.max(0, audioCtx.currentTime - sourceStartCtx)
      );
    }
    return playhead;
  }

  function stopSource() {
    if (!sourceNode) return;
    const node = sourceNode;
    sourceNode = null;
    try { node.stop(); } catch {}
    try { node.disconnect(); } catch {}
  }

  async function startFrom(offset) {
    if (!audioReady || !audioBuffer) return;
    const ctx = ensureAudioContext();
    if (ctx.state === "suspended") await ctx.resume();
    stopSource();
    auditionEnd = null;
    const safe = Math.max(
      0,
      Math.min(Math.max(0, audioBuffer.duration - 0.001), Number(offset) || 0)
    );
    const node = ctx.createBufferSource();
    node.buffer = audioBuffer;
    node.connect(ctx.destination);
    // Schedule a hair in the future so the graph and the visual clock start on
    // the same render quantum instead of racing the current callback.
    const when = ctx.currentTime + 0.02;
    node.start(when, safe);
    sourceNode = node;
    sourceStartCtx = when;
    sourceOffset = safe;
    playhead = safe;
    playing = true;
    playButton.textContent = "❚❚";
    node.onended = () => {
      if (sourceNode !== node) return;
      sourceNode = null;
      playing = false;
      playhead = Math.min(audioBuffer.duration, safe + (ctx.currentTime - when));
      playButton.textContent = "▶";
      timeLabel.textContent = formatTime(playhead);
      draw();
    };
    cancelAnimationFrame(playFrame);
    tick();
  }

  function pausePlayback() {
    if (playing) {
      playhead = currentTime();
      playing = false;
      stopSource();
    }
    auditionEnd = null;
    cancelAnimationFrame(playFrame);
    playButton.textContent = "▶";
    timeLabel.textContent = formatTime(playhead);
    draw();
  }

  function seekTo(seconds) {
    const limit = audioBuffer ? audioBuffer.duration : Infinity;
    playhead = Math.max(0, Math.min(limit, Number(seconds) || 0));
    timeLabel.textContent = formatTime(playhead);
    if (playing) {
      startFrom(playhead).catch(() => {});
    } else {
      draw();
    }
  }

  function tick() {
    if (!playing) {
      playButton.textContent = "▶";
      cancelAnimationFrame(playFrame);
      draw();
      return;
    }
    const time = currentTime();
    timeLabel.textContent = formatTime(time);
    if (auditionEnd !== null && time >= auditionEnd) {
      pausePlayback();
      return;
    }
    if (time > view.end) {
      const span = view.end - view.start;
      setView(time - span * 0.12, time + span * 0.88);
    }
    draw();
    playFrame = requestAnimationFrame(tick);
  }

  function sizeofFile(file) {
    const mb = file.size / (1024 * 1024);
    return `${mb >= 10 ? mb.toFixed(0) : mb.toFixed(1)} MB`;
  }

  function showFile(label, title, meta, input) {
    label.classList.add("has-file");
    title.textContent = input.files[0].name;
    meta.textContent = `${sizeofFile(input.files[0])} · ready`;
    startButton.disabled = !audioInput.files.length;
  }

  audioInput.addEventListener("change", () => {
    if (audioInput.files.length) {
      showFile(audioInput.parentElement, audioTitle, audioMeta, audioInput);
    }
  });
  midiInput.addEventListener("change", () => {
    if (midiInput.files.length) {
      showFile(midiInput.parentElement, midiTitle, midiMeta, midiInput);
    }
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!audioInput.files.length) {
      setStatus("Choose a recording first.");
      return;
    }
    const body = new FormData();
    body.append("audio", audioInput.files[0], audioInput.files[0].name);
    if (midiInput.files.length) {
      body.append("midi", midiInput.files[0], midiInput.files[0].name);
    }
    startButton.disabled = true;
    startButton.classList.add("is-loading");
    setStatus(
      midiInput.files.length
        ? "Opening the session…"
        : `Transcribing locally with MuScriptor ${CONFIG.muscriptorModel || "small"}…`,
      "info"
    );
    try {
      const response = await fetch("/api/label/sessions", {
        method: "POST",
        body,
        headers: { Accept: "application/json" },
      });
      const data = await response.json().catch(() => null);
      if (!response.ok) {
        throw new Error((data && data.error) || `HTTP ${response.status}`);
      }
      await openSession(data);
      setStatus("");
    } catch (error) {
      setStatus((error && error.message) || "The session could not be opened.");
    } finally {
      startButton.disabled = false;
      startButton.classList.remove("is-loading");
    }
  });

  async function openSession(data) {
    session = data;
    stopSource();
    playing = false;
    audioReady = false;
    playButton.disabled = true;
    playhead = 0;
    timeLabel.textContent = formatTime(0);
    groups = data.groups || [];
    notes = groups.flatMap((group) =>
      (group.notes || []).map((note) => ({ ...note, _group: group }))
    );
    noteById = new Map(notes.map((note) => [note.note_id, note]));
    const pitches = notes.map((note) => Number(note.pitch)).filter(Number.isFinite);
    pitchMin = pitches.length ? Math.min(...pitches) - 2 : 40;
    pitchMax = pitches.length ? Math.max(...pitches) + 2 : 80;
    pitchMin = Math.max(0, Math.floor(pitchMin));
    pitchMax = Math.min(127, Math.ceil(pitchMax));
    if (pitchMax - pitchMin < 12) pitchMax = pitchMin + 12;
    const duration = Number(data.duration) || 1;
    view = { start: 0, end: Math.max(1, duration) };
    upload.hidden = true;
    recent.hidden = true;
    workspace.hidden = false;
    empty.hidden = notes.length > 0;
    nameLabel.textContent =
      (data.meta && (data.meta.original_name || data.meta.midi_name)) || data.id;
    exportLink.href = `/api/label/sessions/${data.id}/labels.jsonl`;
    history.replaceState(null, "", `${location.pathname}?session=${encodeURIComponent(data.id)}`);
    updateReviewed();
    updateCounts();
    setSaved(false);
    resizeCanvas();
    draw();
    try {
      await loadAudio(`/api/label/sessions/${data.id}/audio`);
    } catch (error) {
      playButton.disabled = true;
      setStatus("The session audio could not be loaded.");
    }
    if (notes.length) canvas.focus();
  }

  function formatSessionDate(value) {
    if (!value) return "unknown date";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "unknown date";
    return date.toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  }

  function renderSessions(sessions) {
    recentList.replaceChildren();
    if (!sessions.length) {
      recent.hidden = true;
      return;
    }
    for (const item of sessions) {
      const row = document.createElement("div");
      row.className = "label-session-row";

      const button = document.createElement("button");
      button.type = "button";
      button.className = "label-session";
      button.dataset.sessionId = item.id;

      const copy = document.createElement("span");
      const title = document.createElement("span");
      title.className = "label-session-title";
      title.textContent = item.name || item.id;
      const meta = document.createElement("span");
      meta.className = "label-session-meta";
      meta.textContent = `${formatSessionDate(item.created_at)} · ${Number(item.notes) || 0} notes · ${Number(item.dead) || 0} dead`;
      copy.append(title, meta);

      const state = document.createElement("span");
      state.className = "label-session-state";
      if (item.reviewed) {
        state.classList.add("is-reviewed");
        state.textContent = "Reviewed";
      } else {
        state.textContent = "In progress";
      }
      button.append(copy, state);
      button.addEventListener("click", () => openSessionById(item.id));

      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "label-session-delete";
      remove.title = "Delete session";
      remove.setAttribute("aria-label", `Delete ${item.name || item.id}`);
      remove.textContent = "×";
      remove.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        deleteSession(item.id, item.name || item.id);
      });

      row.append(button, remove);
      recentList.append(row);
    }
    recent.hidden = false;
  }

  async function loadSessions() {
    try {
      const response = await fetch("/api/label/sessions", {
        headers: { Accept: "application/json" },
      });
      if (!response.ok) return;
      const data = await response.json();
      renderSessions(Array.isArray(data.sessions) ? data.sessions : []);
    } catch (error) {
      // The upload form still works if the session list cannot be read.
    }
  }

  async function openSessionById(sessionId) {
    if (!sessionId) return;
    setStatus("Opening session…", "info");
    try {
      const response = await fetch(`/api/label/sessions/${encodeURIComponent(sessionId)}`, {
        headers: { Accept: "application/json" },
      });
      const data = await response.json().catch(() => null);
      if (!response.ok) {
        throw new Error((data && data.error) || `HTTP ${response.status}`);
      }
      await openSession(data);
      setStatus("");
    } catch (error) {
      showUpload();
      setStatus((error && error.message) || "That label session could not be opened.");
    }
  }

  async function deleteSession(sessionId, name) {
    if (!sessionId) return;
    const ok = window.confirm(
      `Delete "${name}"?\n\nThis removes the session audio, MIDI and labels from this machine.`
    );
    if (!ok) return;
    try {
      const response = await fetch(`/api/label/sessions/${encodeURIComponent(sessionId)}`, {
        method: "DELETE",
        headers: { Accept: "application/json" },
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const wasCurrent = Boolean(session && session.id === sessionId);
      if (wasCurrent) showUpload();
      else await loadSessions();
      setStatus("Session deleted.", "info");
    } catch (error) {
      setStatus("The session could not be deleted.");
    }
  }

  function showUpload() {
    pausePlayback();
    stopSource();
    audioBuffer = null;
    audioReady = false;
    playhead = 0;
    auditionEnd = null;
    playButton.disabled = true;
    session = null;
    notes = [];
    groups = [];
    noteById = new Map();
    pending = null;
    selection = null;
    undoStack = [];
    workspace.hidden = true;
    upload.hidden = false;
    empty.hidden = true;
    scrollbar.hidden = true;
    history.replaceState(null, "", location.pathname);
    loadSessions();
    setStatus("");
  }

  function updateCounts() {
    const dead = notes.filter((note) => note.label === "dead").length;
    const label = `${notes.length} notes · ${groups.length} groups · ${dead} dead`;
    countsLabel.textContent = label;
  }

  function updateReviewed() {
    const reviewed = Boolean(session && session.meta && session.meta.reviewed);
    reviewButton.textContent = reviewed ? "Reviewed ✓" : "Mark reviewed";
    reviewButton.classList.toggle("is-on", reviewed);
  }

  function timeToX(seconds) {
    const width = Math.max(1, canvas.clientWidth - PAD.left - PAD.right);
    return PAD.left + ((seconds - view.start) / Math.max(1e-9, view.end - view.start)) * width;
  }

  function xToTime(x) {
    const width = Math.max(1, canvas.clientWidth - PAD.left - PAD.right);
    return view.start + ((x - PAD.left) / width) * (view.end - view.start);
  }

  function pitchToY(pitch) {
    const height = Math.max(1, canvas.clientHeight - PAD.top - PAD.bottom);
    const rows = Math.max(1, pitchMax - pitchMin + 1);
    const row = (pitchMax - pitch) / rows;
    return PAD.top + row * height;
  }

  function yToPitch(y) {
    const height = Math.max(1, canvas.clientHeight - PAD.top - PAD.bottom);
    const rows = Math.max(1, pitchMax - pitchMin + 1);
    const ratio = Math.min(0.999, Math.max(0, (y - PAD.top) / height));
    return pitchMax - Math.floor(ratio * rows);
  }

  function noteRect(note) {
    const x0 = timeToX(Number(note.time_s));
    const x1 = Math.max(x0 + 4, timeToX(Number(note.time_s) + Math.max(0.03, Number(note.duration_s) || 0.03)));
    const rowH = Math.max(4, (canvas.clientHeight - PAD.top - PAD.bottom) / Math.max(1, pitchMax - pitchMin + 1));
    const y = pitchToY(Number(note.pitch));
    const h = Math.max(3, Math.min(11, rowH - 1));
    return { x: x0, y: y - h * 0.5, w: Math.max(3, x1 - x0), h };
  }

  function resizeCanvas() {
    const ratio = window.devicePixelRatio || 1;
    const width = wrap.clientWidth;
    const height = wrap.clientHeight;
    canvas.width = Math.max(1, Math.floor(width * ratio));
    canvas.height = Math.max(1, Math.floor(height * ratio));
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    draw();
  }

  function setView(start, end) {
    if (!session) return;
    const duration = Math.max(0.1, Number(session.duration) || end || 1);
    const span = Math.max(0.08, Math.min(duration, end - start));
    const left = Math.max(0, Math.min(start, duration - span));
    view = { start: left, end: left + span };
    if (duration > 0) {
      zoomSlider.value = String(
        Math.max(1, Math.min(12, duration / (view.end - view.start)))
      );
    }
    draw();
  }

  function updateScrollbar() {
    if (!session) {
      scrollbar.hidden = true;
      return;
    }
    const duration = Math.max(0.1, Number(session.duration) || view.end);
    const span = Math.max(0.01, Math.min(duration, view.end - view.start));
    scrollbar.hidden = false;
    const track = scrollbar.clientWidth;
    if (track <= 0) return;
    const thumbWidth = Math.max(28, Math.min(track, track * (span / duration)));
    const maxLeft = Math.max(0, track - thumbWidth);
    const left = duration > 0 ? (view.start / duration) * maxLeft : 0;
    scrollThumb.style.width = `${thumbWidth}px`;
    scrollThumb.style.transform = `translateX(${Math.max(0, Math.min(maxLeft, left))}px)`;
    scrollbar.hidden = false;
  }

  function draw() {
    if (!session) return;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#1f1d19";
    ctx.fillRect(0, 0, width, height);

    // Horizontal pitch guides.
    ctx.font = "10px 'Instrument Sans', sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let pitch = pitchMin; pitch <= pitchMax; pitch += 1) {
      const y = pitchToY(pitch);
      if (pitch % 12 === 0) {
        ctx.strokeStyle = "rgba(246, 241, 232, 0.14)";
        ctx.beginPath();
        ctx.moveTo(PAD.left, y);
        ctx.lineTo(width - PAD.right, y);
        ctx.stroke();
        ctx.fillStyle = "rgba(246, 241, 232, 0.45)";
        ctx.fillText(`E${Math.floor(pitch / 12) - 1}`, PAD.left - 6, y);
      } else if (pitch % 2 === 0) {
        ctx.strokeStyle = "rgba(246, 241, 232, 0.045)";
        ctx.beginPath();
        ctx.moveTo(PAD.left, y);
        ctx.lineTo(width - PAD.right, y);
        ctx.stroke();
      }
    }

    // Beat and bar lines from the MIDI tempo map.
    const beats = (session.beats || []).filter(
      (value) => value >= view.start - 0.01 && value <= view.end + 0.01
    );
    const downbeats = new Set(
      (session.downbeats || []).map((value) => Number(value).toFixed(3))
    );
    for (const beat of beats) {
      const x = timeToX(beat);
      const bar = downbeats.has(Number(beat).toFixed(3));
      ctx.strokeStyle = bar ? "rgba(246, 241, 232, 0.22)" : "rgba(246, 241, 232, 0.08)";
      ctx.beginPath();
      ctx.moveTo(x, PAD.top);
      ctx.lineTo(x, height - PAD.bottom);
      ctx.stroke();
      if (bar) {
        ctx.fillStyle = "rgba(246, 241, 232, 0.42)";
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        ctx.fillText(String(Math.round(beat)), x + 2, 2);
      }
    }

    // Notes.
    for (const note of notes) {
      const rect = noteRect(note);
      if (rect.x + rect.w < PAD.left || rect.x > width - PAD.right) continue;
      const isDead = note.label === "dead";
      ctx.fillStyle = isDead ? "#c2663d" : "#7fa66a";
      if (note === hoverNote) ctx.fillStyle = isDead ? "#e07a4e" : "#9dc282";
      ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
      if (isDead) {
        ctx.strokeStyle = "#2a1812";
        ctx.lineWidth = 1.4;
        const inset = Math.min(3, rect.h * 0.25);
        ctx.beginPath();
        ctx.moveTo(rect.x + inset, rect.y + inset);
        ctx.lineTo(rect.x + rect.w - inset, rect.y + rect.h - inset);
        ctx.moveTo(rect.x + rect.w - inset, rect.y + inset);
        ctx.lineTo(rect.x + inset, rect.y + rect.h - inset);
        ctx.stroke();
      }
      if (note === pending?.note) {
        ctx.strokeStyle = "#f6f1e8";
        ctx.lineWidth = 1;
        ctx.strokeRect(rect.x - 0.5, rect.y - 0.5, rect.w + 1, rect.h + 1);
      }
    }

    if (selection) {
      const x = Math.min(selection.x0, selection.x1);
      const y = Math.min(selection.y0, selection.y1);
      const w = Math.abs(selection.x1 - selection.x0);
      const h = Math.abs(selection.y1 - selection.y0);
      ctx.fillStyle = selection.clear ? "rgba(246, 241, 232, 0.12)" : "rgba(194, 102, 61, 0.22)";
      ctx.fillRect(x, y, w, h);
      ctx.strokeStyle = selection.clear ? "#f6f1e8" : "#e07a4e";
      ctx.setLineDash([5, 4]);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);
    }

    if (playing) {
      const x = timeToX(currentTime());
      if (x >= PAD.left && x <= width - PAD.right) {
        ctx.strokeStyle = "#f6f1e8";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(x, PAD.top);
        ctx.lineTo(x, height - PAD.bottom);
        ctx.stroke();
      }
    }
    updateScrollbar();
  }

  function hitTest(x, y) {
    for (let index = notes.length - 1; index >= 0; index -= 1) {
      const note = notes[index];
      const rect = noteRect(note);
      if (x >= rect.x - 2 && x <= rect.x + rect.w + 2 && y >= rect.y - 2 && y <= rect.y + rect.h + 2) {
        return note;
      }
    }
    return null;
  }

  function pushUndo(changes) {
    if (!changes.length) return;
    undoStack.push(changes);
    if (undoStack.length > 80) undoStack.shift();
  }

  function applyLabels(changes) {
    if (!changes.length) return;
    const previous = changes
      .map((change) => {
        const note = noteById.get(change.note_id);
        return note ? { note_id: change.note_id, label: note.label } : null;
      })
      .filter(Boolean);
    pushUndo(previous);
    for (const change of changes) {
      const note = noteById.get(change.note_id);
      if (note) note.label = change.label;
    }
    updateCounts();
    setSaved(true);
    draw();
    saveLabels(changes);
  }

  async function saveLabels(changes) {
    if (!session) return;
    try {
      const response = await fetch(`/api/label/sessions/${session.id}/labels`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ labels: changes }),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      setSaved(false);
    } catch (error) {
      setSaved(true);
      setStatus("Could not save labels. Check the server.");
    }
  }

  function toggleNote(note) {
    applyLabels([
      { note_id: note.note_id, label: note.label === "dead" ? "pitched" : "dead" },
    ]);
    auditionGroup(note._group);
  }

  function clearAll() {
    const changes = notes
      .filter((note) => note.label === "dead")
      .map((note) => ({ note_id: note.note_id, label: "pitched" }));
    if (!changes.length) return;
    applyLabels(changes);
  }

  function undo() {
    const changes = undoStack.pop();
    if (!changes || !changes.length) return;
    for (const change of changes) {
      const note = noteById.get(change.note_id);
      if (note) note.label = change.label;
    }
    updateCounts();
    setSaved(true);
    draw();
    saveLabels(changes);
  }

  function auditionGroup(group) {
    if (!group || !audioReady) return;
    const start = Math.max(0, Number(group.start_s) || 0);
    startFrom(start)
      .then(() => {
        auditionEnd = Math.max(start + 0.05, Number(group.end_s) || start + 0.2);
      })
      .catch(() => {});
  }

  function seekFromEvent(event) {
    const rect = canvas.getBoundingClientRect();
    seekTo(xToTime(event.clientX - rect.left));
  }

  function beginPointer(event) {
    if (!session) return;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    if (x < PAD.left || x > canvas.clientWidth - PAD.right) return;
    const note = hitTest(x, y);
    canvas.setPointerCapture(event.pointerId);
    if (note && !event.altKey) {
      pending = { note, x0: x, y0: y, moved: false };
    } else {
      selection = { x0: x, y0: y, x1: x, y1: y, clear: Boolean(event.altKey) };
    }
    draw();
  }

  function movePointer(event) {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const note = hitTest(x, y);
    if (note !== hoverNote) {
      hoverNote = note;
      draw();
    }
    if (pending) {
      if (Math.abs(x - pending.x0) + Math.abs(y - pending.y0) > 4) {
        pending.moved = true;
        selection = { x0: pending.x0, y0: pending.y0, x1: x, y1: y, clear: false };
        pending = null;
      }
    }
    if (selection) {
      selection.x1 = x;
      selection.y1 = y;
      draw();
    }
  }

  function endPointer(event) {
    if (pending && !pending.moved) {
      toggleNote(pending.note);
      pending = null;
      draw();
      return;
    }
    if (!selection) {
      pending = null;
      return;
    }
    const x = Math.min(selection.x0, selection.x1);
    const y = Math.min(selection.y0, selection.y1);
    const w = Math.abs(selection.x1 - selection.x0);
    const h = Math.abs(selection.y1 - selection.y0);
    const clear = selection.clear;
    selection = null;
    pending = null;
    if (w < 4 && h < 4) {
      seekFromEvent(event);
      draw();
      return;
    }
    const changes = [];
    for (const note of notes) {
      const rect = noteRect(note);
      const intersects =
        rect.x + rect.w >= x &&
        rect.x <= x + w &&
        rect.y + rect.h >= y &&
        rect.y <= y + h;
      if (!intersects) continue;
      const next = clear ? "pitched" : "dead";
      if (note.label !== next) {
        changes.push({ note_id: note.note_id, label: next });
      }
    }
    applyLabels(changes);
    draw();
  }

  canvas.addEventListener("pointerdown", beginPointer);
  canvas.addEventListener("pointermove", movePointer);
  canvas.addEventListener("pointerup", endPointer);
  canvas.addEventListener("pointercancel", () => {
    pending = null;
    selection = null;
    draw();
  });
  canvas.addEventListener("pointerleave", () => {
    hoverNote = null;
    draw();
  });
  canvas.addEventListener("contextmenu", (event) => {
    event.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const note = hitTest(event.clientX - rect.left, event.clientY - rect.top);
    if (note && note.label === "dead") {
      applyLabels([{ note_id: note.note_id, label: "pitched" }]);
    }
  });

  canvas.addEventListener("wheel", (event) => {
    if (!session) return;
    event.preventDefault();
    const span = view.end - view.start;
    if (event.shiftKey || event.deltaX !== 0) {
      const raw = event.deltaX || event.deltaY;
      // Trackpads emit many small deltas; a wheel notch is roughly 100.
      // One notch moves about 5% of the visible span.
      const direction = raw > 0 ? 1 : -1;
      const step = Math.min(0.10, Math.max(0.01, Math.abs(raw) / 100 * 0.05));
      const delta = direction * step * span;
      setView(view.start + delta, view.end + delta);
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const anchor = xToTime(event.clientX - rect.left);
    const factor = event.deltaY > 0 ? 1.2 : 0.833333;
    zoomAround(anchor, factor);
  }, { passive: false });

  function zoomAround(anchor, factor) {
    if (!session) return;
    const span = view.end - view.start;
    const nextSpan = Math.max(0.08, span * factor);
    const leftShare = (anchor - view.start) / Math.max(1e-9, span);
    const start = anchor - leftShare * nextSpan;
    setView(start, start + nextSpan);
  }

  function zoomBy(factor) {
    if (!session) return;
    zoomAround((view.start + view.end) / 2, factor);
  }

  zoomInButton.addEventListener("click", () => zoomBy(1 / 1.25));
  zoomOutButton.addEventListener("click", () => zoomBy(1.25));

  zoomSlider.addEventListener("input", () => {
    if (!session) return;
    const duration = Number(session.duration) || view.end;
    const span = Math.max(0.08, duration / Number(zoomSlider.value || 1));
    zoomAround((view.start + view.end) / 2, span / Math.max(0.08, view.end - view.start));
  });

  fitButton.addEventListener("click", () => {
    if (!session) return;
    setView(0, Math.max(1, Number(session.duration) || 1));
    zoomSlider.value = "1";
  });

  let scrollDrag = null;
  scrollThumb.addEventListener("pointerdown", (event) => {
    if (!session) return;
    event.preventDefault();
    event.stopPropagation();
    scrollThumb.setPointerCapture(event.pointerId);
    scrollDrag = {
      pointerX: event.clientX,
      startView: view.start,
      span: view.end - view.start,
      duration: Math.max(0.1, Number(session.duration) || view.end),
      track: scrollbar.clientWidth,
      thumb: scrollThumb.getBoundingClientRect().width,
    };
  });

  scrollThumb.addEventListener("pointermove", (event) => {
    if (!scrollDrag) return;
    const maxLeft = Math.max(1, scrollDrag.track - scrollDrag.thumb);
    const ratio = (event.clientX - scrollDrag.pointerX) / maxLeft;
    const delta = ratio * Math.max(0, scrollDrag.duration - scrollDrag.span);
    const start = scrollDrag.startView + delta;
    setView(start, start + scrollDrag.span);
  });

  function endScrollDrag() {
    scrollDrag = null;
  }
  scrollThumb.addEventListener("pointerup", endScrollDrag);
  scrollThumb.addEventListener("pointercancel", endScrollDrag);

  scrollbar.addEventListener("pointerdown", (event) => {
    if (!session || event.target === scrollThumb) return;
    const rect = scrollbar.getBoundingClientRect();
    const duration = Math.max(0.1, Number(session.duration) || view.end);
    const span = view.end - view.start;
    const ratio = (event.clientX - rect.left) / Math.max(1, rect.width);
    const center = Math.max(0, Math.min(duration, ratio * duration));
    setView(center - span / 2, center + span / 2);
  });

  playButton.addEventListener("click", () => {
    if (!session || !audioReady) return;
    if (playing) {
      pausePlayback();
      return;
    }
    const time = currentTime();
    const start = time < view.start || time > view.end ? view.start : time;
    startFrom(start).catch(() => {});
  });

  undoButton.addEventListener("click", undo);
  clearButton.addEventListener("click", clearAll);
  newButton.addEventListener("click", showUpload);
  deleteButton.addEventListener("click", () => {
    if (!session) return;
    deleteSession(
      session.id,
      (session.meta && (session.meta.original_name || session.meta.midi_name)) || session.id
    );
  });
  reviewButton.addEventListener("click", async () => {
    if (!session) return;
    const reviewed = !(session.meta && session.meta.reviewed);
    reviewButton.disabled = true;
    try {
      const response = await fetch(`/api/label/sessions/${session.id}/reviewed`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ reviewed }),
      });
      const data = await response.json().catch(() => null);
      if (!response.ok) throw new Error((data && data.error) || `HTTP ${response.status}`);
      session.meta = data.meta || session.meta;
      session.meta.reviewed = reviewed;
      updateReviewed();
    } catch (error) {
      setStatus("Could not update the reviewed flag.");
    } finally {
      reviewButton.disabled = false;
    }
  });

  window.addEventListener("keydown", (event) => {
    if (!session) return;
    if (event.target instanceof HTMLInputElement) return;
    if (event.code === "Space") {
      event.preventDefault();
      playButton.click();
    } else if (event.key === "u" || event.key === "U") {
      undo();
    } else if (event.key === "Escape") {
      selection = null;
      pending = null;
      draw();
    }
  });

  const observer = new ResizeObserver(() => resizeCanvas());
  observer.observe(wrap);
  window.addEventListener("resize", resizeCanvas);
  resizeCanvas();

  const requestedSession = new URLSearchParams(location.search).get("session");
  loadSessions();
  if (requestedSession) {
    openSessionById(requestedSession);
  }
})();
