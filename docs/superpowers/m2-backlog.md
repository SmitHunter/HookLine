# M2 backlog (carried over from M1 reviews)

## Do first (gates dead-air removal and captions)
- **whisper.cpp DTW timestamps.** M1 runs without `--dtw`, so word times are interpolated: real pauses don't show up as word gaps, and cut points can land up to ~0.3s inside speech. Add `--dtw large.v3.turbo`, parse the per-token `t_dtw` fields (verify the format), add a test that real pauses become word gaps, and nudge each cut point to the nearest low-energy frame within ±250 ms. Replace Task 8's transcript-only boundary check with an audio-based one.

## Clip quality (seen in the M1 real run)
- Word-level cold opens (a clause or the tail of a sentence, cap ~5s). M1 only keeps whole sentences ≤3.5s, and the median real sentence is 4.9s.
- Trim leading connectives at word level ("so", "and", "but", "I mean", "you know"). Needs DTW.
- A real-transcript fixture (CC-BY slice, attributed) for selection tests. The synthetic 2.3s sentences hid the cold-open problem.
- Keep the extra candidates (clips+3) so Swap moment can reuse them without another Claude call.

## Pipeline
- Selection cache key (clips, model, min/max) in picks.json; re-select on mismatch.
- Make the cut a workspace intermediate so the spec's single final encode holds.
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
