# Hookline

Hookline turns a long video, such as a stream VOD, podcast or interview, into 5 short vertical clips. It downloads the video (or takes a local file), transcribes it with Whisper (GPU-accelerated whisper.cpp with word-level DTW timestamps where available, faster-whisper otherwise), has Claude pick the 5 best self-contained moments, and renders each one with ffmpeg into a 1080x1920 MP4 plus a JSON sidecar (title, hook text, post caption, hashtags, rubric scores, layout, keep ranges, removed seconds).

**Status:** M2.5: the CLI pipeline produces captioned, loudness-normalised vertical clips with post copy, and `hookline audit` checks every render against a measurable quality bar. There is no web UI yet, and no Remotion/HyperFrames animated edits yet.

## What a run does

1. **Ingest and transcribe.** Download (yt-dlp) or take a local file; transcribe with word-level timestamps. For Twitch VODs, also download chat.
2. **Signals.** One low-resolution decode of the whole VOD gives per-second motion, visual flashes, audio onsets and game-audio transients (scares), and static screens (death / loading / menu screens and fades: near-uniform or near-white frames that are also grey, with a stream's facecam masked out). Claude sees loudness peaks, chat spikes, scares and death-screen markers on the transcript.
3. **Selection.** Claude proposes candidates with a rubric score, a verdict (`post` / `maybe` / `skip`), a hook type and the clip's first words. Hookline snaps them to sentence boundaries and can add:
   - a **cold open** (a short payoff line played first, then an `EARLIER` card where the clip proper starts),
   - **drops** (lines removed from inside the clip),
   - an extended ending when a scare lands just after it.

   It keeps one clip per encounter on a stream and never splices two attempts across a death screen. A separate **packaging pass** writes the title, hook text, post caption (with a question), 3–5 hashtags and a search title, checked against the clip's own words, plus **caption fixes** for misheard words.
4. **Edit.** Long pauses are cut (dead-air removal) and every cut lands in silence. Wordless stretches are trimmed: over 6 s on talk sources, 3 s on gameplay, keeping the part nearest the next line or scare. Static and death screens inside a clip are cut, or covered with the full facecam when the streamer talks over them.
5. **Layout.** Each scene is reframed to 9:16:
   - **STACKED**: facecam over gameplay. On streams, the facecam is detected once per VOD (`layout_profile.json`) and every clip uses the same fixed template. Short full-cam **reaction beats** show the streamer's face, kept clear of scares and gameplay spikes, and near-black gameplay is lifted.
   - **FACE_TRACK**: follows the speaker. An edge guard keeps the face off the crop edge, the crop leaves room in the direction the speaker faces (gaze lead), and zoom steps land as instant cuts.
   - **SPLIT**: two people, one above the other.
   - **FIT_BLUR**: the full frame over a blurred background.

   Burned-in overlays (lower-thirds, edge widgets) are cropped away unless `--keep-overlays` is given.
6. **Render.** Word-by-word captions are anchored to the voice, so they start with the voice and are not held past it, and a 3 s hook title is burned in. Audio is normalised to −14 LUFS (true peak ≤ −1 dBTP), measured on the speech, so game noise or silence does not skew the level. There is one lossy video encode and one lossy audio encode. Each clip gets a JSON sidecar with everything the audit needs.

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
- `--keep-overlays`: keep burned-in lower-thirds and edge overlays (no overlay crop). Use it when a full-width HUD panel is being cropped as if it were an overlay.

Clips are 20–50 s by default. Every setting can also be set with a `HOOKLINE_<NAME>` environment variable, for example `HOOKLINE_CLIPS`, `HOOKLINE_MIN_CLIP_S` / `HOOKLINE_MAX_CLIP_S`, `HOOKLINE_LLM_BACKEND`, `HOOKLINE_SELECT_MODEL`, `HOOKLINE_TRANSCRIBER` (`auto`, `faster-whisper`, `whisper-cpp`), `HOOKLINE_WHISPER_DEVICE` (whisper.cpp Vulkan device index), `HOOKLINE_COOKIES_FROM_BROWSER` (passed to yt-dlp), `HOOKLINE_MAX_PAUSE_S` and `HOOKLINE_FACECAM`.

The first vertical run downloads a small face-detection model, and for Twitch VODs also TwitchDownloaderCLI, into `tools/`.

### Requirements (`hookline doctor`)

Doctor prints one row per requirement and exits 1 when a required one is missing:

- **Required:** ffmpeg 7 or newer, with libass (the subtitles filter); yt-dlp; the bundled Anton font; valid settings.
- **Optional:** the `claude` CLI (needed for the default `claude-code` backend); whisper.cpp with DTW timestamps (falls back to faster-whisper); a discrete GPU visible to Vulkan; the h264_nvenc / h264_amf encoders; CUDA; the face-detection model and TwitchDownloaderCLI (both downloaded on first use).

## Quality bar (`hookline audit`)

```
uv run hookline audit <job_id>
```

This command re-measures every clip in `output/<job_id>/` against its sidecar and the ASS file it was rendered with. It prints a table per clip, writes `output/<job_id>/audit.json`, and exits 1 if any check fails. A WARN never fails a clip. It flags something to look at.

| Check | Measures | Threshold | Fails or warns |
|---|---|---|---|
| Q1 | Integrated loudness | −14 ±1 LUFS, true peak ≤ −1 dBTP | FAIL |
| Q2 | Caption size, measured on rendered frames | cap height ≥ 55 px at 1080×1920 | FAIL |
| Q3 | Caption coverage | ≥ 95% of spoken time has a caption on screen | FAIL |
| Q3b | Voiced audio with no transcribed word | no stretch > 0.3 s | WARN |
| Q4 | Safe zones (caption and hook boxes) | x in [60, 888], y in [290, 1250] | FAIL |
| Q4b | Captions, hook and tag over the face | ≤ 10% overlap with the face core | FAIL |
| Q5 | Hook title | ≤ 2 lines, ≤ 7 words, cap height ≥ 60 px, on screen 0–3 s (or until the cold-open seam) | FAIL |
| Q6 | Opening | speech by 0.5 s; the first words match the pick's declared first words (≥ 60%) | FAIL (WARN when the opener guard moved the start) |
| Q7 | Pacing | a visible framing change (cut, zoom step, shift, beat or cover edge) at least every 5 s | FAIL |
| Q8 | Encode | 1080×1920 H.264 yuv420p; video ≥ 5 Mb/s, or 2.5× the source bitrate for soft sources (floor 2 Mb/s); AAC 192k | FAIL |
| Q-static | Voiceless static, death or loading screen frames on screen | ≤ 0.5 s | FAIL |
| Q-voiceless | Longest stretch without a caption word, not counting scare windows | ≤ the source's limit (6 s talk, 3 s gameplay) | WARN |
| Q-seam | Audio step or pop at the cold-open seam | ≤ 12 dB rise per 5 ms above the source's own rise | FAIL |
| Q-beat | Full-cam beats over a scare (−0.5 s / +1 s) or a gameplay spike | none | FAIL |

Two extra rows can appear. ASS fails when the recorded ASS file is missing or its hash changed. ERR fails when the audit itself crashed on that clip.

Q-static uses the same screen decision as the cut. When the job's `signals.json` is present, a frame counts only if its source frame lies in a screen run the pipeline detected, so dark gameplay is not flagged as a static screen.

Framing (no false STACKED, no soft panel), selection quality and packaging accuracy are not automated. Review them by watching the clips.

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

Run `hookline doctor` to confirm the whisper.cpp row is OK. Doctor also has a "GPU (whisper.cpp)" row: it runs `vulkaninfo --summary` (when installed) and lists the Vulkan devices, flagging "discrete GPU not visible — check driver" when no discrete GPU shows up.

During transcription Hookline logs the Vulkan device whisper.cpp runs on and warns if it is an integrated GPU (`uma`), which usually means the discrete GPU's driver failed to load. whisper.cpp uses device 0 by default; set `HOOKLINE_WHISPER_DEVICE=<n>` to pick another one (passed as `-dev <n>`; the warning suggests the index of the first discrete device).

| Backend | Hardware | Speed (`large-v3-turbo`) |
|---|---|---|
| whisper.cpp Vulkan | discrete AMD GPU | ~39x realtime (10 min audio in 15.5 s) |
| faster-whisper int8 | CPU | ~2x realtime (10 min audio in 294 s) |

Measured on synthetic repeated TTS speech, so real streams with denser speech will be somewhat slower.

## Limitations

- **English only.** Selection prompts, caption fixes and the Anton caption font assume English (Latin script). Other languages transcribe, but captions and packaging are not tuned for them.
- **Game voice-over is captioned like the streamer.** There is no speaker separation, so in-game dialogue is transcribed and captioned as speech.
- **Animated or pulsing facecam borders** are not handled. The cam is profiled as a static rectangle, and only one cam per stream is masked (no duo cams).
- **SPLIT shows everyone detected**, including people who are not speaking. There is no active-speaker detection.
- Static-screen detection is tuned on horror streams. Very dark, colourless gameplay can still be cut as a screen, and very colourful death screens are not detected.
- The gameplay crop can cut off on-screen hands or weapons at the panel edges.
