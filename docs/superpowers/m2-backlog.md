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
- h264_nvenc quality args are untested (no NVIDIA GPU here).

## New in M2 (from the real run and reviews)
- False STACKED on wide multi-person shots: the podcast's wide table shot got a zoomed, soft facecam crop on top of the wide frame (seen in clips 01 and 04). Needs a size/aspect sanity check on the detected face box, or a multi-face veto.
- Choppiness from many short keep ranges (clip 01 has 7, clip 03 has 10 in under a minute); add a minimum gap to cut and merge nearby cuts.
- VFR sources may drift by up to a frame per cut.
- Horizontal + captions is untested.
- One ffmpeg spawn per face sample; batch frame extraction.
- Layout flicker on fast-cut gameplay is likely; add scene-layout hysteresis.
- Twitch VOD real-world check: pending (no URL provided).
- Transcription with DTW took about 41 min for the 1h+ podcast on the Vulkan GPU, far slower than the M1 estimate; measure and consider chunking or a lighter DTW preset.
- The first hook title box overlaps the top of the head on tight FACE_TRACK crops; consider placing it higher or smaller.
- Source lower-third graphics are cropped mid-word in FACE_TRACK and STACKED crops.
- Clip 03 uses a cold open, so keep_ranges are not monotonic (the opening line repeats later); intended, but the sidecar consumer should expect it.
- `output/<job>/` accumulates old M1 clips alongside new ones when a job is re-run; clean the folder or version it.
