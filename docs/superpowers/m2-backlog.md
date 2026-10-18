# M2 backlog (carried over from M1 reviews)

## Do first (gates dead-air removal and captions)
- ~~**whisper.cpp DTW timestamps.**~~ Done in M2.
  Original note: M1 runs without `--dtw`, so word times are interpolated: real pauses don't show up as word gaps, and cut points can land up to ~0.3s inside speech. Add `--dtw large.v3.turbo`, parse the per-token `t_dtw` fields (verify the format), add a test that real pauses become word gaps, and nudge each cut point to the nearest low-energy frame within ±250 ms. Replace Task 8's transcript-only boundary check with an audio-based one.

## Clip quality (seen in the M1 real run)
- ~~Word-level cold opens~~ Done in M2. Original note: (a clause or the tail of a sentence, cap ~5s). M1 only keeps whole sentences ≤3.5s, and the median real sentence is 4.9s.
- ~~Trim leading connectives~~ Done in M2. Original note: at word level ("so", "and", "but", "I mean", "you know"). Needs DTW.
- A real-transcript fixture (CC-BY slice, attributed) for selection tests. The synthetic 2.3s sentences hid the cold-open problem.
- Keep the extra candidates (clips+3) so Swap moment can reuse them without another Claude call.

## Pipeline
- ~~Selection cache key~~ Done in M2. Original note: (clips, model, min/max) in picks.json; re-select on mismatch.
- ~~Make the cut a workspace intermediate~~ Done in M2 (single final encode). Original note: so the spec's single final encode holds.
- Chunked, resumable transcription for 4–8h VODs; whisper.cpp `--vad`.
- Download progress, and resumable downloads (download.tmp is wiped on every run).
- Route select.py logging into run_job's `log` callback (needed for M3's live progress).
- Doctor: check the claude CLI is logged in, not just on PATH.

## Polish / test gaps
- `repo_root()` should return None when no repo marker is found; `test_repo_root_points_at_repo` depends on the checkout layout.
- `_backend_label` uses type names; give backends a `label` attribute.
- Tests: `FFmpegError` truncation, the `pick_encoder` fallback, an output path with spaces, `WhisperCppBackend.transcribe` with a fake binary, the CUDA branch, apostrophes in paths (speech_wav fixture), and the vacuous assertion in `test_ingest_url_uses_ytdlp`.
- whisper.cpp malformed entries raise a raw KeyError; the `av<18` pin needs a comment; tests import `httpx2` directly.
- Build script: sort Vulkan SDK folders as versions, and support non-default SDK paths.
- Chunk mode: one failed chunk fails the whole selection; overlap can exceed the budget slightly.
- h264_nvenc quality args are untested (no NVIDIA hardware in CI or local testing).

## New in M2 (from the real run and reviews)
- ~~False STACKED on wide multi-person shots~~ Fixed in the M2 final review: multi-face veto, 160 px minimum cam box, and small-face group shots go to FIT_BLUR. Original note: the podcast's wide table shot got a zoomed, soft facecam crop on top of the wide frame (seen in clips 01 and 04).
- ~~Choppiness from many short keep ranges~~ Addressed in the M2 final review (max pause 0.6 s, pad 0.12 s, no kept piece under 1.5 s); re-check on the next real run. Original note: clip 01 has 7, clip 03 has 10 in under a minute.
- VFR sources may drift by up to a frame per cut. (Cut points are now snapped to the source's frame grid; VFR itself is untested.)
- Horizontal + captions test: the `--horizontal` path with captions burned in has no test.
- One ffmpeg spawn per face sample; batch frame extraction.
- ~~Layout flicker on fast-cut gameplay is likely~~ Addressed for streams in M2.5: the facecam is profiled once per VOD and every clip uses one fixed STACKED template. Non-stream sources still choose a layout per scene without hysteresis.
- ~~Twitch VOD real-world check~~ Done in M2.5 (horror-game stream runs drove the stream features and the Q-static / Q-voiceless / Q-seam / Q-beat checks).
- Transcription of a 1 h+ podcast took about 41 min when whisper.cpp fell back to an integrated GPU (`uma: 1`) because the discrete GPU's driver had not loaded. DTW is not the cost. The run now logs the device and warns on `uma`; `HOOKLINE_WHISPER_DEVICE` selects a device and `hookline doctor` reports the GPU. Re-measure on a working discrete GPU.
- ~~The first hook title box overlaps the top of the head on tight FACE_TRACK crops~~ Fixed in the M2 final review: `hook_y_for` moves the hook above the captions when the first scene's face reaches the hook band (STACKED: under the seam captions).
- ~~Source lower-third graphics are cropped mid-word in FACE_TRACK and STACKED crops.~~ Done in M2.5: overlay bands are detected and cropped away (`--keep-overlays` opts out).
- Clip 03 uses a cold open, so keep_ranges are not monotonic (the opening line repeats later); intended, but the sidecar consumer should expect it.
- clips.json manifest + stale output cleanup: `output/<job>/` accumulates old clips alongside new ones when a job is re-run; write a manifest of the current clips and remove outputs it no longer lists.
- Hybrid per-clip DTW (fast full pass without DTW, DTW only over the picked clips) only if a discrete-GPU re-measure shows DTW costs more than 3x.
- Chunked/resumable full-pass transcription, so a crash or driver fault mid-VOD does not lose the whole pass.
- ~~Overlay band detection (lower-thirds, Twitch widgets/alerts)~~ Done in M2.5 (static edge bands; a stream's cam label is masked). Bordered, text-only full-width HUD panels can still be cropped as overlays (use `--keep-overlays`).
- Collapsed DTW word starts (several words sharing one start) → spread them evenly across the span instead of stacking caption words.
- Mark the slow ffmpeg tests (`pytest -m slow`): the full suite now takes about 15 minutes.

## Left over from the M2 final review
- SPLIT: the hook title can cover the bottom speaker's face when faces sit near the top of their panel; place it in the seam gap instead.
- A hand-edited transcript.json with a non-dict `key` raises AttributeError (not treated as corrupt).
- Chunked selection assumes one line per sentence (strict zip); a newline inside word text would raise.
- ~~Cache migration~~ Done in M2.5: old-format transcript.json files with `timing == "dtw"` are accepted when the backend matches.
- Hook and caption placement use only the first scene's layout (Q4b now catches captions over a later scene's face).

## Done in M2.5 (quality pass)
- Two-pass, speech-gated loudness normalisation to −14 LUFS / −1 dBTP with a single lossy audio encode.
- Caption and hook sizing, coverage, safe zones; voice-anchored caption timing; captions and hook kept off the face.
- Selection: verdicts, hook types, declared first words, cold opens, drops, opener guard, payoff extension, a packaging pass (post caption, hashtags, search title) and caption fixes.
- Pacing: zoom steps and framing changes at most 5 s apart; FACE_TRACK edge guard and gaze lead.
- Stream editing: VOD-level facecam profile and fixed STACKED template, full-cam reaction beats, covers over talked-over death screens, scare/flash signals, static and death screens cut, voiceless trims.
- `hookline audit` and the automated quality bar (Q1–Q8, Q3b, Q4b, Q-static, Q-voiceless, Q-seam, Q-beat); doctor requires ffmpeg ≥ 7.

## Open after M2.5

### Features (M3+)
- Emoji pop-ins, sound effects, keyword pop-ups and a progress bar (animated edits).
- Speaker separation: game voice-over is transcribed and captioned like the streamer.
- Active-speaker detection, so SPLIT does not show people who are not talking.
- A content-category enum for `diversify` (it currently keys on free-text categories).
- Calm talking-head sources give flat clips; selection cannot add energy the source lacks.

### Stream layout
- Animated or pulsing facecam borders are not handled (the cam is profiled as a static rectangle).
- Second / duo cam masking: only one cam per stream is profiled and masked.
- The gameplay crop can slice hands or held items at the panel edges.
- `JC_FACE_MIN` (0.12) may treat a large in-game face as a Just Chatting face when the cam is hidden.
- Static-overlay detection samples at about 1 Hz and can suppress punch-ins on still gameplay.
- A borderless cam over a textured wall relaxes the HUD check.
- Merge framing changes that land a few milliseconds apart (a hold start and a beat pulse).

### Selection and boundaries
- An untrimmed fallback skips the opener guard; an extended teaser can start on "and"/"so".
- `_cold_open` treats "..." as a sentence end.
- A straight single quote in a title regex can mangle apostrophes.
- Answer-token openers ("No one…", "Sure enough…") should require a comma or sentence end after the token; edge pieces are measured from the line start, not the clip start.
- Drop logs are emitted for clips later rejected by duration; mid-line strip messages are logged before a fallback.
- `_hard_max` raises the limit for length caps under 20 s (document or change).
- A stutter shift can push a short second word under the minimum word length.

### Render and captions
- Captions on tight close-ups can sit over hair or the upper forehead.
- A short opening punch-in can release early when an edge-guard hold starts; guard pulses ease at about 300 px/s.
- Sub-80 ms words away from cuts remain.
- Q4b's face box is not dilated by the caption outline.
- A stray `.graph.txt` filter script can be left behind when a render is interrupted.

### Loudness
- Limiter attack is 1 ms; try 3–5 ms. Clamp the re-aimed gain.
- Tests for `plan_loudness` branches (first-pass break, non-converging to dynamic, skip/None) and a NaN guard on parsed numbers; the +20 dB gain cap is only string-tested.

### Chat
- `_chat_hot` is computed twice; stretched emotes (e.g. "KEKWWW") are not matched.

### Tests
- Tests to tighten: the front-trim weak-start test is vacuous, the FACE_TRACK shift test never asserts a shift happens, letterbox-offset facecams are untested end to end.
