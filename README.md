# Hookline

Hookline turns a long video, such as a stream VOD, podcast or interview, into 5 short vertical clips. It downloads the video (or takes a local file), transcribes it with Whisper (GPU-accelerated whisper.cpp with word-level DTW timestamps where available, faster-whisper otherwise), has Claude pick the 5 best self-contained moments, and renders each one with ffmpeg into a 1080x1920 MP4 plus a JSON sidecar (title, hook text, reason, rubric scores, layout, keep ranges, removed seconds).

**Status:** M2: the CLI pipeline produces captioned vertical clips. There is no web UI yet, and no Remotion/HyperFrames animated edits yet.

Features:

- Vertical 9:16 reframing, chosen per scene: STACKED (facecam over gameplay, including Twitch facecam stacking), FACE_TRACK (follows the speaker), SPLIT (two people side by side), and FIT_BLUR (full frame over a blurred background).
- Burned-in word-by-word captions and a 3 s hook title at the start of each clip.
- Dead-air removal: long pauses inside a clip are cut out.
- Silence-snapped cuts: clip boundaries move to the nearest low-energy point in the audio.
- Chat- and loudness-aware picks: Claude also sees audio loudness peaks and, for Twitch VODs, chat activity.

## Quickstart

1. Install [uv](https://docs.astral.sh/uv/) and [ffmpeg](https://ffmpeg.org/) (both on your `PATH`), plus the [`claude` CLI](https://docs.claude.com/en/docs/claude-code) signed in to your account (the default selection backend).
2. Install dependencies and check the setup:

```
cd apps/server
uv sync
uv run hookline doctor
```

3. Cut clips from a URL or a local file:

```
uv run hookline run <url-or-file>
```

Without the GPU whisper.cpp build, the first run downloads the Whisper model (~1.6 GB) and transcribes on CPU at about 2x realtime. `workspace/` and `output/` are created in the current directory.

Clips land in `output/<job_id>/NN-<slug>.mp4` with a matching `.json`. Intermediate artifacts (download, audio, transcript, picks) are cached in `workspace/<job_id>/`, so re-running the same source skips finished stages and only re-cuts. Options: `--clips N`, `--backend claude-code|api`, `--model`, `--out DIR`, plus:

- `--horizontal`: keep the source aspect ratio instead of reframing to 9:16.
- `--no-captions`: do not burn in captions or the hook title.
- `--no-chat`: skip the Twitch chat download and chat signals.
- `--facecam X,Y,W,H`: facecam box as 0-1 fractions of the frame, overriding auto-detection.

The first vertical run downloads a small face-detection model, and for Twitch VODs also TwitchDownloaderCLI, into `tools/`.

## How clips are picked

Claude reads the timestamped transcript and proposes candidate moments, each scored on a 100-point rubric: Hook 25 · Standalone context 15 · Payoff 15 · Emotional intensity 15 · Length fit 10 · Pacing 10 · Loop potential 5 · Platform safety 5. Hookline then snaps every clip to sentence boundaries (so it starts and ends on a complete sentence), keeps clips within the length limits, avoids overlaps, and keeps the top 5 by total score.

## Copyright

This tool processes media you have the right to use. Clipping other creators' streams or copyrighted films without a license may infringe copyright and lead to DMCA takedowns, strikes or demonetization. In the US, fair use is decided case by case: commentary and transformation help, but plain re-uploads rarely qualify. Australia has no general fair use, only narrower fair dealing purposes (criticism/review with acknowledgement, parody/satire, news), and moral rights still apply. Get written permission from creators and campaign owners, follow platform policies, and treat licensed music separately. Not legal advice.

## Claude backends

Clip selection runs through the `claude` CLI by default (`HOOKLINE_LLM_BACKEND=claude-code`) or the Anthropic API (`api`). The API backend enables server-side refusal fallback by default (`fallbacks="default"`).

## AMD / Intel GPU on Windows

whisper.cpp publishes no Windows Vulkan binary, so build it once (installs CMake and the Vulkan SDK via winget, builds `whisper-cli.exe`, downloads `ggml-large-v3-turbo`):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build-whisper-vulkan.ps1
```

Hookline then picks whisper.cpp automatically: when `HOOKLINE_WHISPER_CPP_BIN` and `HOOKLINE_WHISPER_CPP_MODEL` are unset, it looks for `<repo>/tools/bin/whisper-cli.exe` and `<repo>/tools/models/ggml-large-v3-turbo.bin` (`whisper-cli` on non-Windows) and uses them if they exist. To use a different build or model, set those two variables to your own paths. With neither the variables nor those files it falls back to faster-whisper. `hookline run` prints which backend it is using.

Run `hookline doctor` to confirm the whisper.cpp row is OK.

| Backend | Hardware | Speed (`large-v3-turbo`) |
|---|---|---|
| whisper.cpp Vulkan | AMD Radeon RX 7800 XT | ~39x realtime (10 min audio in 15.5 s) |
| faster-whisper int8 | CPU | ~2x realtime (10 min audio in 294 s) |

Measured on synthetic repeated TTS speech, so real streams with denser speech will be somewhat slower.
