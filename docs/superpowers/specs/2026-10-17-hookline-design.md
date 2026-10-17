# Hookline — AI clip factory (design + build plan)

> Working name **Hookline** is my call; rename freely before the GitHub repo is created.

## Context

One app: drop in any long video (Twitch VODs/clips, podcasts, movies, other creators' content) → get 5 short-form vertical clips that look professionally edited (OpusClip/Submagic quality or better).

Primary use case: a Twitch streamer's VODs first, then other long-form creators. So v1 must nail Twitch, but the architecture must not block batch processing later.

Why build instead of buy (from research): OpusClip-style tools charge per-minute credits, watermark free tiers, cut mid-sentence, and are weak on gaming / facecam layouts. **No open-source tool handles Twitch facecam + gameplay stacking well** — that is this app's edge.

## Product decisions
| Decision | Answer |
|---|---|
| Claude access | **Both, switchable**: Claude Code CLI (`claude -p`, uses the user's Claude subscription) default; Anthropic API key option |
| App shape | **Local web app now** (browser at localhost), Tauri desktop wrapper later |
| Renderer | **Both, split by job**: Remotion = fixed templates (captions, zooms, progress bar, SFX, final composite); HyperFrames = Claude-authored one-off overlays (hook cards, callouts) |
| Stream type to handle | **Mix**: gameplay+facecam AND Just Chatting scenes in the same VOD → per-scene layout detection |
| Highlight picking | Claude reads the **whole transcript** (annotated with heat signals) and picks 5 |
| Input mix | **Mostly full VODs (hours long)**, some short clips. Input < ~2 min → skip selection, single-clip "polish" mode (reframe + trim + caption + edit) |
| Context budget | Opus does selection on the full transcript in one pass; chunked map-reduce only when transcript exceeds the context budget (rare) |
| Stream layout | No sample yet → build generic corner-facecam detection + drag-to-fix box in UI; tune on a real VOD later |
| Repo | Clean codebase + great README for GitHub |

## Goals / non-goals / anti-goals
- **Goals:** any source (file, Twitch VOD/clip URL, YouTube URL) → 5 clips, each with a real hook in the first 3s, dead air removed, 9:16 reframed, animated captions, high-end motion graphics. Nice GUI. Clean, documented codebase.
- **Non-goals (v1):** auto-posting to TikTok/YT/IG, hosted multi-user service, Tauri wrapper, brand kits / campaign batches, multi-account variants. (All listed in README roadmap as v2.)
- **Anti-goals (fail even if it "works"):** clips that start or end mid-sentence; captions out of sync after dead-air cuts; facecam cropped off or wrong face tracked; install that needs an afternoon of dependency fixing; credits/watermarks.

## Hardware constraint (reference dev machine)
AMD Ryzen 7 7700X, **AMD RX 7800 XT (no NVIDIA/CUDA)**, 31 GB RAM. Installed: ffmpeg 8.1 (gyan full build, has libass + h264_amf), Python 3.13, Node 24, uv, pnpm, gh, claude CLI. Missing: yt-dlp (becomes a Python dependency).
→ Transcription default = **whisper.cpp Vulkan build** (fast on AMD). faster-whisper (CUDA / CPU int8) as alternate backend for NVIDIA users on GitHub. Encoder: `h264_amf` when present, `libx264` fallback.

## Architecture

```
hookline/
├─ apps/
│  ├─ server/          Python 3.12 (uv), FastAPI, SQLite (SQLModel) — the pipeline
│  │  └─ hookline/
│  │     ├─ api/            REST + SSE progress
│  │     ├─ jobs/           job runner (asyncio worker, resumable stages)
│  │     ├─ pipeline/       ingest, transcribe, signals, select, cut, reframe, edit_plan, render, export
│  │     ├─ llm/            ClaudeBackend protocol → ClaudeCodeBackend | AnthropicApiBackend
│  │     ├─ transcribe/     TranscriberBackend → WhisperCppBackend | FasterWhisperBackend
│  │     ├─ models/         pydantic schemas (Transcript, Signals, ClipPick, EditDecisionList)
│  │     └─ tools/          binary discovery/download (whisper.cpp, TwitchDownloaderCLI)
│  └─ web/             Vite + React + TS + Tailwind + shadcn/ui + TanStack Query
├─ packages/
│  ├─ render-remotion/      Remotion project: <Short> composition driven by EDL JSON props
│  ├─ render-hyperframes/   HyperFrames project: overlay templates + Claude-authored compositions
│  └─ schema/               JSON Schema for EDL (generated from pydantic) → TS types
├─ assets/                  OFL fonts, CC0 SFX pack, caption style presets
├─ scripts/                 setup (doctor + binary downloads), dev runner
├─ docs/                    architecture.md, pipeline.md, contributing.md
└─ README.md
```

Each pipeline stage writes its artifact into `workspace/<job_id>/` and is skipped if the artifact exists → reruns/retries are cheap, and any clip can be regenerated from the review screen without re-transcribing.

## Pipeline (stage by stage)

1. **ingest** — local file or URL via `yt-dlp` (Twitch VOD/clip, YouTube, etc.). Twitch VOD → also fetch chat JSON with **TwitchDownloaderCLI** (MIT, auto-downloaded by `tools/`). Extract 16 kHz mono WAV. Run `cropdetect` once to find letterbox/black bars ("crop dead space").
2. **transcribe** — word-level timestamps. whisper.cpp (Vulkan, `large-v3-turbo`, DTW token timestamps) default; faster-whisper alternate. Auto-picks backend from detected hardware; overridable in Settings. Output: `transcript.json` (segments + words).
3. **signals** — per-second "heat" timeline:
   - chat velocity vs rolling median (flag ≥2–3×), emote counts (KEKW, LUL, OMEGALUL, Pog…)
   - audio loudness peaks (ffmpeg `ebur128`)
   - scene boundaries (PySceneDetect)
4. **select (Claude)** — full transcript compacted to `[mm:ss] sentence` lines with inline heat markers (`🔥chat×4`, `🔊loud`). Prompt uses the scoring rubric below and returns **structured JSON** (schema-validated, retried once on failure): 5 clips, each with `start`, `end` (snapped to sentence boundaries), optional `cold_open` (a ≤3s line from later in the clip moved to the front if the natural opening is weak), `hook_text` (on-screen), `title`, per-criterion scores + total, `reason`, `edit_beats` (emphasis words, zoom moments, emoji moments). Overlapping picks deduped (>50% overlap). Transcripts exceeding the context budget → chunked map (candidates per chunk) then reduce (final 5).
   - Rubric (0–100, from research): Hook 25 · Standalone context 15 · Payoff 15 · Emotional intensity (incl. heat) 15 · Length fit 10 · Pacing 10 · Loop potential 5 · Platform safety 5.
   - Length: 20–60s target, Claude decides per clip (my call; research says 20–45s usually wins, 60s only if tension holds).
5. **cut** — ffmpeg trim; **dead-air removal** by cutting word gaps > 0.4s (keep 0.08s padding) plus silero-vad for music/game-audio-heavy clips; rebuild a **remapped word timeline** for the cut clip (this keeps captions in sync — anti-goal).
6. **reframe** — per scene (from PySceneDetect) classify layout:
   - `STACKED` — facecam detected in a stable corner box + gameplay → facecam top ⅓, gameplay crop bottom ⅔ (`crop` + `vstack`)
   - `FACE_TRACK` — one dominant face (Just Chatting, podcasts) → 9:16 crop following the face, smoothed (MediaPipe face detection + exponential smoothing)
   - `SPLIT` — two speakers → stacked halves
   - `FIT_BLUR` — fallback: fit + blurred background
   Facecam box auto-detected once per VOD from sampled frames; **user can drag-correct it in the UI** and it's saved per channel. Reframe logic adapted from **OpenShorts** (MIT, credited in README/NOTICE).
7. **edit_plan (Claude)** — per clip, Claude emits an **EditDecisionList** (EDL) JSON: caption style preset, word emphasis/colour, zoom punch-ins (timestamps + scale), emoji pop-ins, SFX cues (whoosh on cuts, pop on emoji), progress bar on/off, hook card spec, optional callout specs. Validated against shared schema.
8. **render** —
   - **HyperFrames** renders the hook card + callouts as **transparent WebM** overlays. Claude writes them from a curated template set (`render-hyperframes/templates/`), output passes `hyperframes lint`; on lint/render failure fall back to the stock template with Claude's text. Pinned HyperFrames version (pre-1.0).
   - **Remotion** renders the final 1080×1920 MP4: base reframed clip (`@remotion/media` `<Video>`) + TikTok-style word captions (`@remotion/captions`) + zooms + emoji + progress bar + SFX + HyperFrames overlays as video layers. One final encode.
   - **Quick mode** (fallback + fast previews): ffmpeg + ASS karaoke captions only. Also used if Node/Chrome render fails.
9. **export** — `output/<job>/<NN>-<slug>.mp4` + `clip.json` (title, hook, suggested post caption + hashtags, scores, reason).

## Claude backends (`llm/`)
- `ClaudeCodeBackend`: runs `claude -p --output-format json` with the prompt + schema; no API key needed.
- `AnthropicApiBackend`: official `anthropic` SDK, structured output via tool use; key from Settings/env `ANTHROPIC_API_KEY`.
- Default model `claude-opus-5-5` for selection (quality matters most), `claude-sonnet-5-5` for EDL/overlay authoring; both configurable. (Load `claude-api` skill when implementing.)

## Web UI (`apps/web`)
Dark, video-tool aesthetic (use `frontend-design` skill when building):
1. **Home** — big drop zone + URL paste, recent jobs.
2. **Job** — stage timeline with live SSE progress/logs, ETA.
3. **Review** — 5 clip cards: player, total score + rubric breakdown, Claude's reason, hook text (editable), trim handles, caption-style picker, facecam box correction, "Regenerate clip" / "Re-render", export.
4. **Settings** — Claude backend + key, transcriber backend/model, output folder, default caption style/fonts/colours.

## Licensing / legal (README)
- Repo **MIT**. NOTICE credits OpenShorts (MIT) for adapted reframe code. Do **not** copy SupoClip code (AGPL).
- Remotion is source-available: free for individuals / companies ≤3 people; README states forks inside 4+ person companies need a Remotion licence. HyperFrames is Apache-2.0.
- Copyright note: tool is for media you have rights to; clipping others/movies needs permission; US fair use vs AU fair dealing; not legal advice.

## Build order (milestones — each ends with something visible)
| # | Milestone | Visible win |
|---|---|---|
| M1 | Repo skeleton, `scripts/doctor`, ingest, transcribe (whisper.cpp Vulkan), Claude select, raw cuts; **CLI** `hookline run <file|url>` | 5 raw clips + reasons from a real VOD |
| M2 | Signals (chat/loudness/scenes), dead-air cut + timeline remap, 4 reframe layouts, Quick-mode ASS captions | Postable vertical captioned clips |
| M3 | FastAPI + job runner + SQLite, web UI (Home/Job/Review/Settings) | Drag-drop in browser → clips |
| M4 | Remotion `<Short>` composition + Claude EDL (captions, zooms, emoji, SFX, progress bar) | "Highly edited" look |
| M5 | HyperFrames hook cards + callouts → composited in Remotion | Custom motion graphics per clip |
| M6 | README (hero GIF, mermaid diagram, quickstart, GPU table, licence/legal), docs/, cleanup pass, `/simplify` + code review | Repo ready to push |

Rough agent build time: M1–M2 ≈ one long session each; M3–M5 ≈ one session each; M6 ≈ half a session. After approval, the next step is `superpowers:writing-plans` to turn each milestone into a TDD task list.

## Verification
- **Unit (pytest):** dead-air gap cutting + word-timeline remap, chat spike detection, layout geometry math, transcript compaction, Claude JSON parsing/validation against fixtures, EDL schema round-trip (pydantic ⇄ TS).
- **Integration:** short fixture video committed in `tests/fixtures/` (CC0 or self-generated) → full pipeline in Quick mode with a stub Claude backend returning canned picks; assert 5 MP4s at 1080×1920, durations within bounds, captions present.
- **Render smoke:** Remotion and HyperFrames each render one 5s sample from a fixture EDL.
- **Real-world check (manual):** run a real Twitch VOD end-to-end via the UI; review the 5 clips for hook strength, sync, facecam framing.
- `scripts/doctor` verifies ffmpeg (libass, h264_amf), whisper.cpp Vulkan, Node/Chrome for renderers, Claude backend reachability.

## Platform + ingest details (my calls)
- **Windows-first**, no Windows-only code. `scripts/doctor` + binary downloads target Windows (Vulkan + CUDA + CPU); macOS/Linux documented as best-effort.
- whisper.cpp Vulkan on Windows: use an official prebuilt release asset if one exists; otherwise doctor offers a build-from-source step (Vulkan SDK + CMake), and until then falls back to faster-whisper CPU int8 so the pipeline never blocks.
- Sub-only / login-gated VODs: Settings toggle → `yt-dlp --cookies-from-browser <browser>`.
- Long VODs: transcription is chunked (e.g. 10-min segments, parallelised) with resumable progress so a 4–8h VOD survives restarts.

## Review actions (my call)
- **Swap moment** — re-run selection with the current pick excluded → new clip.
- **Re-edit** — keep the moment; re-run edit_plan + render only (optionally with a user note, e.g. "more zooms").
- **Re-render** — apply manual tweaks (hook text, trim, caption style, facecam box) without calling Claude.

## Builder's calls
- Name "Hookline".
- Python 3.12 via uv (MediaPipe/whisper wheels lag on 3.13).
- Clip length 20–60s chosen per clip, not fixed 60s.
- Optional "cold open" reordering to guarantee a 3s hook.
- CLI-first in M1 so the core pipeline is proven before UI work.
- Bundled CC0 SFX + OFL fonts only (safe for open source and client work).
