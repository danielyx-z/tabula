# Tabula

### Why?
I'd frequently listen to guitar covers on Youtube but many would have paid or no sheets, so I built this to help me transcribe audio into guitar tabs.

### Comparison Against Leading Models (Songscription, Muscriptor, Klang.io)
**Original Audio:** *Lover Girl* by Laufey, played by matthewnjee on YouTube

https://github.com/user-attachments/assets/7d3c8f6d-d76f-4b57-ab0d-2ddc98687e3a

**Tabula**

(your Tabula video here)

**Songscription**

https://github.com/user-attachments/assets/74406efc-1eb8-42cc-bd11-878d48d8fe02

**Klang.io**

https://github.com/user-attachments/assets/c6ee543f-7d3c-4c04-9379-67cf58fa3828

**MuScriptor** (no sheet playback feature)

<img width="1483" height="1009" alt="muscriptorimage" src="https://github.com/user-attachments/assets/a12ca3ce-4700-4a97-8605-7d526d0b1bd7" />


Overall, Songscription is the best out of the available models, but it is not tailored for guitar tablature. Only Tabula correctly identifies the percussive dead-notes at the beginning of the song. Klang.io does an okay job, but it strips away some complexity (and sound-good-ability) of the original song, making it boring to listen to. Muscriptor is the base model used in part of Tabula to derive MIDI from audio, but it's outputted tablature offers very little practical use.


### How?
A small T5 encoder-decoder (2.79M params) reads symbolic MIDI note events (`NOTE_ON`, `NOTE_OFF`, `TIME_SHIFT`) and writes tablature events (`TAB`, `TIME_SHIFT`). A local web app wraps the model: download audio from Youtube with yd-dlp, transcribe audio with MuScriptor, label dead notes from the recording, export Guitar Pro 5 scores.

**Pipeline:** audio → MuScriptor → MIDI → T5 → tab events → playability pass (string assignment, capo, collision fixing) → audio onset analysis (dead notes, articulations) → export (.gp5, .mid, .wav, ASCII tab).

Time is quantised to a 20-tick grid at PPQ 960. `TIME_SHIFT` is compositional with 11 values (20–3840 ticks). 


Training corpus: GuitarSet (360 tracks, 62k notes, split by performance).

A separate XGBoost model trained on my own data is used for guitar-specific tab corrections: notably, dead (muted) notes are changed to a percussive X in the output tablature.  

Based on **Fretting-Transformer: Encoder-Decoder Model for MIDI to Tablature Transcription** (Hamberger et al., arXiv:2506.14223, ICMC 2025).

See `docs/` for format specs, data contract, labelling workflow, cluster notes, and verification.
