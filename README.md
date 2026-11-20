# Tabula

I'd frequently listen to guitar covers on Youtube but many would have paid or no sheets, so I built this to help me transcribe audio into guitar tabs.

### Comparison Against Leading Models (Songscription (paid), Klang.io (paid), Muscriptor (free))
**Original Audio:** *Lover Girl* by Laufey, played by matthewnjee on YouTube

https://github.com/user-attachments/assets/1c53f549-2176-48d3-8766-60d2aff74caf

**Tabula** Identifies dead notes correctly

https://github.com/user-attachments/assets/b8bcb013-845f-485a-9496-07aff17de8b7

**Songscription** Fails to identify dead notes

https://github.com/user-attachments/assets/58a79516-ab12-4169-a119-5717edb3888e

**Klang.io** Fails to identify dead notes and excessively simplifies the tab

https://github.com/user-attachments/assets/0290b27f-cfa7-4730-ad33-2c6129cc495e


**MuScriptor** Output is illegible

<img width="1483" height="1009" alt="muscriptorimage" src="https://github.com/user-attachments/assets/a12ca3ce-4700-4a97-8605-7d526d0b1bd7" />


Overall, Songscription is the best out of the available models, but it is not tailored for guitar tablature. Only Tabula correctly identifies the percussive dead-notes at the beginning of the song. 

Klang.io transcription quality is technically correct, but it strips away the detail of the original song, making it boring to listen to. Muscriptor is the base model used in part of Tabula to derive MIDI from audio, but its outputted tablature offers very little practical use.


### How does it work?
A small T5 encoder-decoder (2.79M params) reads symbolic MIDI note events (`NOTE_ON`, `NOTE_OFF`, `TIME_SHIFT`) and writes tablature events (`TAB`, `TIME_SHIFT`). A local web app wraps the model: download audio from Youtube with yd-dlp, transcribe audio with MuScriptor, label dead notes from the recording, export Guitar Pro 5 scores.

**Pipeline:** audio → MuScriptor → MIDI → T5 → tab events → playability pass (string assignment, capo, collision fixing) → audio onset analysis (dead notes, articulations) → export (.gp5, .mid, .wav, ASCII tab).

Time is quantised to a 20-tick grid at PPQ 960. `TIME_SHIFT` is compositional with 11 values (20–3840 ticks). 


### Other notes
Training corpus: GuitarSet (360 tracks, 62k notes, split by performance).

A separate XGBoost model trained on my own data is used for guitar-specific tab corrections: notably, dead (muted) notes are changed to a percussive X in the output tablature.  

See `docs/` for format specs, data contract, labelling workflow, cluster notes, and verification.

This project is based on the paper **Fretting-Transformer: Encoder-Decoder Model for MIDI to Tablature Transcription**.


