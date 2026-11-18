# Tabula

### Why?
I'd frequently listen to guitar covers on Youtube but many would have paid or no sheets, so I built this to help me transcribe audio into guitar tabs.

### How
A small T5 encoder-decoder (2.79M params) reads symbolic MIDI note events (`NOTE_ON`, `NOTE_OFF`, `TIME_SHIFT`) and writes tablature events (`TAB`, `TIME_SHIFT`). A local web app wraps the model: download audio from Youtube with yd-dlp, transcribe audio with MuScriptor, label dead notes from the recording, export Guitar Pro 5 scores.

**Pipeline:** audio → MuScriptor → MIDI → T5 → tab events → playability pass (string assignment, capo, collision fixing) → audio onset analysis (dead notes, articulations) → export (.gp5, .mid, .wav, ASCII tab).

Time is quantised to a 20-tick grid at PPQ 960. `TIME_SHIFT` is compositional with 11 values (20–3840 ticks). Training corpus: GuitarSet (360 tracks, 62k notes, split by performance).

An XGBoost model trained on my own data is used for guitar-specific tab corrections: notably, dead (muted) notes are changed to a percussive X in the output tablature.  

Based on **Fretting-Transformer: Encoder-Decoder Model for MIDI to Tablature Transcription** (Hamberger et al., arXiv:2506.14223, ICMC 2025).

See `docs/` for format specs, data contract, labelling workflow, cluster notes, and verification.
