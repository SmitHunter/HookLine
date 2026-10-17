# M2 Vertical Captioned Clips Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `hookline run` outputs postable 1080×1920 clips: accurate word timing, dead air removed, cut points in real pauses, stream-aware vertical layouts and burned-in word-by-word captions with a hook title.

**Architecture:** M1's pipeline keeps its stages: ingest → audio → transcribe → **signals** (new) → select. The cut stage becomes a per-clip render:

1. **refine boundaries:** snap the cut points into silences and trim leading filler words.
2. **dead-air keep ranges:** decide which parts of the clip to keep.
3. **near-lossless horizontal intermediate:** cut those ranges from the source.
4. **scenes + faces → layout:** work out how to frame each scene.
5. **one final encode:** reframe filter + ASS captions.

Signals (chat rate, loudness) are annotated into Claude's transcript. All the timing logic is pure functions over `Word`s and `(start, end)` ranges. ffmpeg only executes the result.

**Tech Stack:** Python 3.12 (uv), ffmpeg 8.1 (silencedetect, astats, libass `subtitles`), whisper.cpp v1.9.5 with DTW, TwitchDownloaderCLI 1.56.5, `opencv-python-headless` + YuNet 2023mar ONNX, `scenedetect` 0.7.2, Anton font (OFL).

**Spec:** `docs/superpowers/specs/2026-10-17-hookline-design.md`. Also read `docs/superpowers/m2-backlog.md` for the carry-over items this plan picks up.

## Global Constraints

- Python `>=3.12,<3.13` via `uv` in `apps/server/`. Lint and format with `ruff`. Every subprocess call uses list args, never `shell=True`. All paths use `pathlib`.
- Artifacts are written atomically (`atomic_write_text`, or tmp + `os.replace`).
- Output: **1080×1920**, H.264 via `encoder_args(pick_encoder())`, AAC 192k, `+faststart`. Exactly **one** final lossy encode per clip. The intermediate in `workspace/` is `libx264 -preset veryfast -crf 12`.
- Layouts (spec, verbatim):
  - `STACKED`: facecam top ⅓, gameplay crop bottom ⅔
  - `FACE_TRACK`: 9:16 crop following the face, smoothed
  - `SPLIT`: two speakers, stacked halves
  - `FIT_BLUR`: fit + blurred background
- Dead air: remove pauses longer than **0.4s**, keeping **0.08s** padding on each side. **Ruling:** pauses come from the audio (ffmpeg `silencedetect`), not from word gaps, because word timings are ±0.2s even with DTW. Music- or game-heavy audio with no detectable silence means nothing is cut. VAD is in the backlog.
- Captions: words come from the dead-air-remapped timeline. They must never drift from the audio (spec anti-goal: "captions out of sync after dead-air cuts").
- Bundled assets may only be OFL fonts and MIT/Apache models. Downloaded tools and models go under `<repo>/tools/` (gitignored) via one fetch helper.
- Windows-first. Paths with spaces and emoji must work everywhere, including the `subtitles=` filter.

## Review Focus

1. **Game or music audio with no silences:** dead-air removal keeps the whole clip and boundary refinement leaves the cut points unchanged. No crash, no empty clip. Pinned in Task 3 and Task 4.
2. **No faces at all** (screen share, gameplay without a cam, a scene with a black frame): the layout falls back to `FIT_BLUR`. Pinned in Task 7.
3. **Caption text containing ASS-special characters** (`{`, `}`, `\`), emoji or non-ASCII, rendered from a directory whose name has spaces and an emoji: the text is escaped and the render succeeds. Pinned in Task 9.
4. **Chat download fails** (non-Twitch URL, deleted VOD, offline, TwitchDownloaderCLI missing): the job continues without chat signals and logs one warning. Pinned in Task 6.
5. **Dead-air removal would eat most of a clip** (a long silent stretch inside it): at least `min_keep_s` of speech remains and the clip is never empty. If the result would be shorter than 50% of the original, the original ranges are kept. Pinned in Task 3.

---

## File structure

```
apps/server/hookline/
├─ assets.py                 fetch(url, dst) atomic downloader; tools_dir(); pinned URLs
├─ media/audio.py            rms_per_second, detect_silences
├─ media/reframe.py          layout_filter(...) → ffmpeg filter graph per layout
├─ media/subtitles.py        subtitles_filter(ass_path, fonts_dir) (escaping)
├─ captions/ass.py           group_words, build_ass
├─ vision/faces.py           Face, FaceDetector (YuNet), sample_frames
├─ pipeline/deadair.py       keep_ranges, remap_words
├─ pipeline/boundaries.py    refine_boundaries (silence snap + leading-filler trim)
├─ pipeline/signals.py       Signals, compute_signals, heat_markers
├─ pipeline/chat.py          twitch_vod_id, download_chat, chat_rate
├─ pipeline/layout.py        SceneLayout, classify_scene, plan_layouts
├─ pipeline/render.py        render_clip (intermediate → scenes → layout → final)
assets/fonts/Anton-Regular.ttf + OFL.txt         (repo root, committed)
apps/server/tests/fixtures/podcast_60s.opus, whisper_cpp_dtw_sample.json, portrait.jpg, ATTRIBUTION.md
```

Modified: `transcribe/whisper_cpp.py`, `transcript_text.py` (markers), `pipeline/select.py` (prompt text and cold-open snapping), `pipeline/run.py`, `pipeline/cut.py` (removed, since `render.py` replaces it), `config.py`, `cli.py`, `doctor.py`, `README.md`.

---

### Task 1: DTW word timestamps + real-audio fixtures

**Files:**
- Modify: `apps/server/hookline/transcribe/whisper_cpp.py`, `apps/server/hookline/models.py`
- Create: `apps/server/tests/fixtures/podcast_60s.opus`, `apps/server/tests/fixtures/whisper_cpp_dtw_sample.json`, `apps/server/tests/fixtures/ATTRIBUTION.md`
- Test: `apps/server/tests/test_transcribe.py`

**Interfaces:**
- Produces:
  - `Transcript.timing: Literal["segment", "dtw", "whisper"] = "segment"`
  - `dtw_preset(model: Path) -> str`, for example `ggml-large-v3-turbo.bin` → `"large.v3.turbo"` and `ggml-base.en.bin` → `"base.en"`. Strip the `ggml-` prefix and the extension, then replace `-` with `.`.
  - `parse_whisper_cpp_json(data, duration) -> Transcript` now prefers DTW timing.

Verified facts:
- whisper.cpp only computes DTW when flash attention is off. Run with `-nfa -dtw <preset>`.
- Each token in `transcription[i].tokens[]` carries `t_dtw` in **10 ms units**. It is `-1` when DTW is unavailable.
- Special tokens have text like `[_TT_232]`.

- [ ] **Step 1: Create the fixtures** from the CC-BY podcast used in M1 ("Interview with Lauren-Glenn Davitian on CCTV and Public Access", Town Meeting TV, https://www.youtube.com/watch?v=9mrYGaeeg04).
  1. `uv run yt-dlp -x --download-sections "*1940-2000" -o tmp.%(ext)s <url>`, then `ffmpeg -i tmp.* -ac 1 -ar 16000 -c:a libopus -b:a 24k tests/fixtures/podcast_60s.opus`.
  2. Produce the DTW JSON: decode the opus to a 16 kHz WAV, then run `tools/bin/whisper-cli.exe -m tools/models/ggml-large-v3-turbo.bin -f <wav> -ojf -of <stem> -ml 1 -sow -nfa -dtw large.v3.turbo` and copy `<stem>.json` to `tests/fixtures/whisper_cpp_dtw_sample.json`.
  3. Write ATTRIBUTION.md (title, channel, URL, "CC BY", the excerpt range).
- [ ] **Step 2: Write the failing tests**

```python
def test_dtw_preset():
    assert dtw_preset(Path("ggml-large-v3-turbo.bin")) == "large.v3.turbo"
    assert dtw_preset(Path("x/ggml-base.en.bin")) == "base.en"

def test_parse_prefers_dtw_and_matches_real_pauses():
    t = parse_whisper_cpp_json(load("whisper_cpp_dtw_sample.json"), duration=60.0)
    words = t.words()
    assert t.timing == "dtw"
    assert 1.0 <= words[0].start <= 1.4            # audio: silence until 1.11s (offsets said 0.00)
    starts = [w.start for w in words]
    assert starts == sorted(starts)                 # monotonic
    assert all(w.end > w.start for w in words)
    after_pause = next(w for w in words if w.start > 5.5)   # silence 5.20–6.22 in the audio
    assert 6.1 <= after_pause.start <= 6.7

def test_parse_falls_back_to_offsets_without_dtw():
    t = parse_whisper_cpp_json(load("whisper_cpp_sample.json"), duration=5.0)   # M1 fixture, t_dtw absent
    assert t.timing == "segment" and t.words()[0].start == pytest.approx(0.32)

def test_backend_passes_dtw_flags(monkeypatch, tmp_path):   # fake Popen records argv
    ...
    assert "-nfa" in argv and argv[argv.index("-dtw") + 1] == "large.v3.turbo"
```

- [ ] **Step 3: Run the tests.** `uv run pytest tests/test_transcribe.py -q` → FAIL
- [ ] **Step 4: Implement.**
  - Word start: `t_dtw / 100` of the entry's first speech token (non-empty text that doesn't start with `[`). Fall back to `offsets.from / 1000`.
  - Then force the starts to be non-decreasing: `start = max(start, previous start)`.
  - Word end: `min(next word's start, last speech token t_dtw/100 + 0.15)`, and at least `start + 0.05`. The last word uses `max(start + 0.05, offsets.to / 1000)`.
  - `timing = "dtw"` when at least 50% of the words had DTW; otherwise use offsets for every word and `timing = "segment"`.
  - faster-whisper sets `timing = "whisper"`.
- [ ] **Step 5: Run the tests**, then the full suite, `ruff check` and `ruff format --check` → PASS
- [ ] **Step 6: Commit.** Message: `feat: DTW word timestamps for whisper.cpp`

---

### Task 2: Audio analysis helpers

**Files:**
- Create: `apps/server/hookline/media/audio.py`
- Test: `apps/server/tests/test_audio.py`

**Interfaces:**
- Produces:
  - `rms_per_second(wav: Path) -> list[float]`: dBFS per 1 s window, with `-inf` clamped to `-100.0`. Uses `asetnsamples=n=16000:p=0,astats=metadata=1:reset=1,ametadata=print:file=-`, parsing `lavfi.astats.Overall.RMS_level` per `pts_time`.
  - `detect_silences(wav: Path, start: float, end: float, noise_db: float = -32.0, min_s: float = 0.35) -> list[tuple[float, float]]`: absolute source times, clipped to `[start, end]`. Uses `-ss start -to end -i wav -af silencedetect=noise=<db>dB:d=<min_s>`. A silence still open at the end closes at `end`.

- [ ] **Step 1: Write the failing tests.** The fixture is built with lavfi: `sine` 1 s, `anullsrc` 1 s, `sine` 2 s, 16 kHz mono.

```python
def test_rms_per_second_marks_silence(tone_gap_wav):
    rms = rms_per_second(tone_gap_wav)
    assert len(rms) == 4 and rms[1] < -60 and rms[0] > -30 and rms[2] > -30

def test_detect_silences_absolute_and_clipped(tone_gap_wav):
    s = detect_silences(tone_gap_wav, 0.5, 4.0)
    assert len(s) == 1 and s[0][0] == pytest.approx(1.0, abs=0.05) and s[0][1] == pytest.approx(2.0, abs=0.05)

def test_detect_silences_none_in_continuous_audio(tone_wav):
    assert detect_silences(tone_wav, 0, 3) == []
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement with `media.ffmpeg._run`, reusing its error handling. **Step 4:** Run → PASS, plus the full suite and ruff.
- [ ] **Step 5: Commit.** Message: `feat: per-second loudness and silence detection`

---

### Task 3: Dead-air keep ranges + word remap

**Files:**
- Create: `apps/server/hookline/pipeline/deadair.py`
- Test: `apps/server/tests/test_deadair.py`

**Interfaces:**
- Consumes: `Word` (models)
- Produces:
  - `keep_ranges(segments: list[tuple[float,float]], silences: list[tuple[float,float]], max_pause: float = 0.4, pad: float = 0.08, min_keep_ratio: float = 0.5) -> list[tuple[float,float]]`
    - For each segment, every silence fully inside it that is longer than `max_pause` is cut out, keeping `pad` on each side.
    - Ranges shorter than 0.05 s are dropped.
    - If the kept total is below `min_keep_ratio` × the original total, it returns `segments` unchanged.
  - `remap_words(words: list[Word], keep: list[tuple[float,float]]) -> list[Word]`
    - Maps each word into the output timeline, where output time = time elapsed through the kept ranges.
    - A word is clipped to the kept ranges it overlaps, and a word entirely inside a removed range is dropped.
    - Ranges may repeat source time (cold open), so a word inside both segments appears twice, at both output positions.

- [ ] **Step 1: Write the failing tests**

```python
def test_cuts_long_pause_keeps_padding():
    assert keep_ranges([(10, 20)], [(12, 13)]) == [(10, 12.08), (12.92, 20)]

def test_short_pause_kept():
    assert keep_ranges([(10, 20)], [(12, 12.3)]) == [(10, 20)]

def test_no_silences_is_identity():
    assert keep_ranges([(10, 20)], []) == [(10, 20)]

def test_refuses_to_eat_most_of_clip():
    assert keep_ranges([(0, 10)], [(1, 9)]) == [(0, 10)]

def test_cold_open_segments_processed_independently():
    assert keep_ranges([(50, 52), (30, 55)], [(40, 41)]) == [(50, 52), (30, 40.08), (40.92, 55)]

def test_remap_shifts_words_after_cut():
    keep = [(10, 12.08), (12.92, 20)]
    out = remap_words([w(11, 11.5, "a"), w(13, 13.4, "b")], keep)
    assert [(x.start, x.end) for x in out] == [(1.0, 1.5), pytest.approx((2.16, 2.56))]

def test_remap_duplicates_words_in_cold_open():
    out = remap_words([w(50.5, 51, "x")], [(50, 52), (30, 55)])
    assert [round(x.start, 2) for x in out] == [0.5, 22.5]
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS, plus the full suite and ruff.
- [ ] **Step 5: Commit.** Message: `feat: dead-air keep ranges and word timeline remap`

---

### Task 4: Boundary refinement + word-level cold opens

**Files:**
- Create: `apps/server/hookline/pipeline/boundaries.py`
- Modify: `apps/server/hookline/pipeline/select.py` (`_cold_open`: snap to word boundaries, cap 5.0 s; update the SYSTEM_PROMPT cold-open wording to "a short phrase or sentence of at most 5 seconds")
- Test: `apps/server/tests/test_boundaries.py`, `apps/server/tests/test_select.py`

**Interfaces:**
- Consumes: `ClipPick`, `Word`; silences from `detect_silences` (Task 2)
- Produces:
  - `FILLER_OPENERS: tuple[str, ...] = ("and", "so", "but", "because", "like", "um", "uh", "yeah", "okay", "well", "i mean", "you know", "anyway")`
  - `refine_boundaries(pick: ClipPick, words: list[Word], silences: list[tuple[float,float]], window: float = 0.4) -> ClipPick`, applied in this order:
    1. **Leading-filler trim.** While the clip's first words (comparing lowercased text with punctuation stripped, for 1–2 word phrases) match `FILLER_OPENERS`, move `start` to the next word's start. Stop at 3 words, or if the clip would drop below 15 s.
    2. **Silence snap.** If a silence overlaps `[start - window, start + window]`, set `start = silence.end - 0.05`. If a silence overlaps `[end - window, end + window]`, set `end = silence.start + 0.05`. With no silence nearby, leave it unchanged.
    3. Never move `start` past the first remaining word's start, nor `end` before the last word's end.
  - Cold opens snap to word boundaries: start = start of the first word starting at or after `co_start - 0.2`; end = end of the last word ending at or before `co_end + 0.2`. Cap 5.0 s, still inside the clip.

- [ ] **Step 1: Write the failing tests**

```python
def test_trims_leading_filler():
    p = refine_boundaries(pick(10, 40), words_from("and so the city never told us", start=10), [])
    assert p.start == word_start("the")

def test_snaps_start_into_preceding_silence():
    p = refine_boundaries(pick(10.2, 40), words_from("the city", start=10.2), [(9.5, 10.1)])
    assert p.start == pytest.approx(10.05)

def test_no_silence_leaves_bounds():
    p = refine_boundaries(pick(10, 40), words_from("the city", start=10), [])
    assert (p.start, p.end) == (10, 40)

def test_cold_open_snaps_to_words_and_allows_5s(...):   # in test_select.py
    ...
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS, plus the full suite and ruff.
- [ ] **Step 5: Commit.** Message: `feat: silence-snapped clip boundaries, filler trim, word-level cold opens`

---

### Task 5: Signals + heat markers in Claude's transcript

**Files:**
- Create: `apps/server/hookline/pipeline/signals.py`
- Modify: `apps/server/hookline/transcript_text.py` (`compact_transcript(sentences, markers: dict[int, list[str]] | None = None)`), `apps/server/hookline/pipeline/select.py` (`select_clips(..., markers=None)` passed through; SYSTEM_PROMPT explains the markers)
- Test: `apps/server/tests/test_signals.py`, `apps/server/tests/test_transcript_text.py`

**Interfaces:**
- Produces:
  - `Signals(BaseModel)`: `duration: float`, `loudness: list[float]` (dB per second), `chat: list[float] | None` (weighted messages per second)
  - `heat_markers(signals: Signals) -> dict[int, list[str]]`:
    - `"🔥chat×N"` where the 10 s rolling chat sum is at least 2.5× the rolling median of the surrounding 300 s (minimum baseline 1.0). N is the rounded ratio. Mark only the first second of each run.
    - `"🔊loud"` where a second's loudness is at least `median + 9 dB` and above −30 dB. Again only the first second of each run.
  - In `compact_transcript`, each line gets the markers whose second falls in `[sentence.start, sentence.end)`, appended in order as ` ⟨🔥chat×4 🔊loud⟩`.
- SYSTEM_PROMPT: add one paragraph saying the markers are audience and audio reactions. They are evidence for emotional intensity, not a requirement.

- [ ] **Step 1: Write the failing tests**

```python
def test_chat_spike_marked_once():
    chat = [1.0] * 600; chat[300:310] = [8.0] * 10
    m = heat_markers(Signals(duration=600, loudness=[-25.0] * 600, chat=chat))
    keys = [k for k, v in m.items() if any(x.startswith("🔥chat") for x in v)]
    assert len(keys) == 1 and 295 <= keys[0] <= 301

def test_loud_marked_and_no_chat_ok():
    loud = [-30.0] * 100; loud[50] = -15.0
    m = heat_markers(Signals(duration=100, loudness=loud, chat=None))
    assert m == {50: ["🔊loud"]}

def test_compact_appends_markers():
    out = compact_transcript([Sentence(start=50, end=53, text="No way.")], {51: ["🔊loud"]})
    assert out == "[00:50.0] No way. ⟨🔊loud⟩"
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS, plus the full suite and ruff.
- [ ] **Step 5: Commit.** Message: `feat: loudness/chat heat markers in the selection transcript`

---

### Task 6: Tool fetcher + Twitch chat

**Files:**
- Create: `apps/server/hookline/assets.py`, `apps/server/hookline/pipeline/chat.py`
- Test: `apps/server/tests/test_assets.py`, `apps/server/tests/test_chat.py`, `apps/server/tests/fixtures/twitch_chat_sample.json` (hand-written, about 20 comments, in the schema below)

**Interfaces:**
- Produces:
  - `tools_dir() -> Path` (`repo_root() / "tools"`)
  - `fetch(url: str, dst: Path) -> Path`: if `dst` exists, return it. Otherwise download to `dst.with_suffix(dst.suffix + ".part")` and `os.replace`. Raise `AssetError` on HTTP failure. Uses `urllib.request`.
  - `ensure_twitch_downloader() -> Path`: fetches `https://github.com/lay295/TwitchDownloader/releases/download/1.56.5/TwitchDownloaderCLI-1.56.5-Windows-x64.zip` to `tools/downloads/`, extracts `TwitchDownloaderCLI.exe` to `tools/bin/`, and returns its path. Windows only; other platforms raise `AssetError("TwitchDownloaderCLI auto-install is Windows-only")`.
  - `twitch_vod_id(url: str) -> str | None`: matches `twitch.tv/videos/<digits>`.
  - `download_chat(vod_id: str, dst: Path, exe: Path) -> Path`: runs `[exe, "chatdownload", "--id", vod_id, "-o", tmp]`, where tmp is `dst` with `.part.json`, then `os.replace`. Raises `ChatError` on a non-zero exit.
  - `chat_rate(chat_json: dict, duration: float) -> list[float]`: per second, 1.0 per comment plus 0.5 per emote fragment (`message.fragments[].emoticon` not null). The time is `comments[].content_offset_seconds`.
  - run_job: when ingesting a Twitch VOD URL, try `ensure_twitch_downloader` + `download_chat` into `job_dir/chat.json`. On `AssetError`, `ChatError` or `OSError`, log `warning: chat unavailable (<reason>) — continuing without chat signals`. This wiring is done in Task 10, but `chat.py` exposes `try_fetch_chat(source: str, job_dir: Path, log) -> Path | None` here.

- [ ] **Step 1: Write the failing tests**

```python
def test_twitch_vod_id():
    assert twitch_vod_id("https://www.twitch.tv/videos/2280112233?t=1h") == "2280112233"
    assert twitch_vod_id("https://youtube.com/watch?v=x") is None

def test_chat_rate_weights_emotes():
    r = chat_rate(load("twitch_chat_sample.json"), duration=10)
    assert len(r) == 10 and r[3] == pytest.approx(2.5)   # 2 comments + 1 emote at second 3 in fixture

def test_try_fetch_chat_non_twitch_returns_none(tmp_path):
    assert try_fetch_chat("https://youtube.com/watch?v=x", tmp_path, log=print) is None

def test_try_fetch_chat_failure_warns_and_continues(tmp_path, monkeypatch):
    monkeypatch.setattr(chat, "ensure_twitch_downloader", raise_(AssetError("offline")))
    logs = []
    assert try_fetch_chat("https://www.twitch.tv/videos/1", tmp_path, logs.append) is None
    assert any("chat unavailable" in l for l in logs)

def test_fetch_is_atomic_and_cached(tmp_path, local_http_server):
    ...   # serves bytes; second call does not re-request; no .part left behind
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS, plus the full suite and ruff.
- [ ] **Step 5: Smoke test** (needs network): `uv run python -c "from hookline.assets import ensure_twitch_downloader as e; print(e())"` → prints a path to `TwitchDownloaderCLI.exe`.
- [ ] **Step 6: Commit.** Message: `feat: tool fetcher and Twitch chat download`

---

### Task 7: Face detection + layout classification

**Files:**
- Create: `apps/server/hookline/vision/__init__.py`, `apps/server/hookline/vision/faces.py`, `apps/server/hookline/pipeline/layout.py`
- Modify: `apps/server/pyproject.toml` (add `opencv-python-headless`, `scenedetect`), `apps/server/hookline/assets.py` (`ensure_yunet() -> Path`)
- Test: `apps/server/tests/test_layout.py`, `apps/server/tests/test_faces.py`, `apps/server/tests/fixtures/portrait.jpg` (a public-domain portrait photo from Wikimedia Commons, source URL recorded in ATTRIBUTION.md)

**Interfaces:**
- Produces:
  - `Face(x: float, y: float, w: float, h: float, score: float)`: normalized 0..1 relative to the frame.
  - `ensure_yunet()` fetches `https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx` into `tools/models/`.
  - `FaceDetector(model: Path, score_threshold: float = 0.6)` with `.detect(bgr: np.ndarray) -> list[Face]`. Uses `cv2.FaceDetectorYN.create(model, "", (w, h), score_threshold, 0.3, 5000)`, calling `setInputSize` per frame size. Columns 0–3 are x, y, w, h and column 14 is the score.
  - `sample_frames(video: Path, times: list[float], width: int = 640) -> list[np.ndarray]`: an ffmpeg `-ss t -i video -frames:v 1 -vf scale=640:-2 -f rawvideo -pix_fmt bgr24 -` per time.
  - `LayoutKind = Literal["STACKED", "FACE_TRACK", "SPLIT", "FIT_BLUR"]`
  - `SceneLayout(BaseModel)`: `start: float`, `end: float` (clip-relative), `kind: LayoutKind`, `facecam: tuple[float,float,float,float] | None` (normalized x, y, w, h), `track_x: list[tuple[float, float]]` (time, normalized centre x keyframes), `split_x: tuple[float, float] | None`
  - `classify_scene(samples: list[tuple[float, list[Face]]], start: float, end: float, facecam_override: tuple[float,float,float,float] | None = None) -> SceneLayout`. Rules, in order:
    1. If `facecam_override` is set → STACKED with that box.
    2. Facecam: in at least 60% of samples there is a face with `h < 0.22` whose centre lies in a corner region (centre x < 0.35 or > 0.65, and centre y < 0.4 or > 0.6), with a centre std below 0.03 → STACKED. The box is the face box grown to 3.0× width and 2.6× height around the face centre, clamped to the frame.
    3. Two faces with `h ≥ 0.15` in at least 50% of samples → SPLIT, with `split_x` = their median centre x values, sorted.
    4. One face in at least 50% of samples → FACE_TRACK, with `track_x` = per-sample centre x of the largest face.
    5. Otherwise → FIT_BLUR.
  - `plan_layouts(video: Path, detector: FaceDetector, facecam_override=None, sample_every: float = 1.0) -> list[SceneLayout]`: scenes from `scenedetect.detect(str(video), ContentDetector())`, where an empty result means one scene spanning the whole clip. Samples at `sample_every` within each scene, then classifies.

- [ ] **Step 1: Write the failing tests**

```python
def test_corner_cam_is_stacked():
    cam = Face(x=0.78, y=0.05, w=0.08, h=0.14, score=0.9)
    s = classify_scene([(t, [cam]) for t in range(10)], 0, 10)
    assert s.kind == "STACKED" and s.facecam[0] < 0.78 and s.facecam[0] + s.facecam[2] <= 1.0

def test_big_face_tracks():
    assert classify_scene([(t, [Face(x=0.4, y=0.2, w=0.2, h=0.4, score=0.9)]) for t in range(10)], 0, 10).kind == "FACE_TRACK"

def test_two_faces_split():
    f1, f2 = Face(x=0.1, y=0.2, w=0.15, h=0.3, score=0.9), Face(x=0.7, y=0.2, w=0.15, h=0.3, score=0.9)
    assert classify_scene([(t, [f1, f2]) for t in range(10)], 0, 10).kind == "SPLIT"

def test_no_faces_fit_blur():
    assert classify_scene([(t, []) for t in range(10)], 0, 10).kind == "FIT_BLUR"

def test_override_forces_stacked():
    assert classify_scene([(0, [])], 0, 5, facecam_override=(0, 0, 0.3, 0.3)).kind == "STACKED"

@pytest.mark.integration
def test_detector_finds_face_in_synthetic_stream_frame(tmp_path):
    # ffmpeg: testsrc2 1920x1080 background + portrait.jpg scaled to 400px tall overlaid at top-right
    faces = FaceDetector(ensure_yunet()).detect(frame)
    assert faces and faces[0].x > 0.6 and faces[0].y < 0.4
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS, plus the full suite and ruff.
- [ ] **Step 5: Commit.** Message: `feat: YuNet face detection and per-scene layout classification`

---

### Task 8: Reframe filter graphs

**Files:**
- Create: `apps/server/hookline/media/reframe.py`
- Test: `apps/server/tests/test_reframe.py`

**Interfaces:**
- Consumes: `SceneLayout` (Task 7)
- Produces:
  - `OUT_W, OUT_H = 1080, 1920`
  - `scene_filter(layout: SceneLayout, in_w: int, in_h: int, label_in: str, label_out: str) -> str`: a filter chain from `[label_in]` (already trimmed to the scene) to `[label_out]` at 1080×1920 / SAR 1.
    - **STACKED:** the facecam crop scales to 1080×640 (scale+crop to fill). Gameplay is a centre crop with aspect 1080:1280 taken from the full frame, scaled to 1080×1280. The two are combined with `vstack`.
    - **FACE_TRACK:** crop width `in_h*9/16`, full height. `x` is a piecewise-linear expression over the smoothed keyframes:
      - EMA α 0.3, with a deadband: the target only moves when the change is greater than 0.05.
      - Clamp to `[0, in_w - crop_w]`.
      - Built as nested `if(lt(t,T1), lerp, ...)`.
      - Then scale to 1080×1920.
    - **SPLIT:** two crops of aspect 1080:960 centred on `split_x[0]` and `split_x[1]`, each scaled to 1080×960, combined with `vstack`.
    - **FIT_BLUR:** `split` into bg and fg. The bg is scaled to cover 1080×1920, cropped and blurred (`boxblur=20:2`); the fg is scaled to width 1080 and overlaid centred.
  - `detect_crop(video: Path) -> tuple[int, int, int, int] | None`: removes black bars ("crop any dead space"). Runs `cropdetect=limit=24:round=2` over 5 frames sampled across the clip and takes the most common `crop=w:h:x:y`. Returns None when that crop removes less than 2% of width and height.
  - `reframe_graph(layouts: list[SceneLayout], in_w: int, in_h: int, crop: tuple[int,int,int,int] | None = None) -> str`: a full `-filter_complex` that does `[0:v]` (`crop=w:h:x:y` first, when given; `in_w`/`in_h` then mean the cropped size) `split=N` → `trim=start:end,setpts=PTS-STARTPTS` per scene → `scene_filter` → `concat=n=N:v=1:a=0[vout]`.

- [ ] **Step 1: Write the failing tests.** `test_detect_crop_letterbox`: a 1280×720 testsrc2 padded with 100 px black bars top and bottom (`pad=1280:920:0:100`) → crop ≈ (1280, 720, 0, 100); an unpadded input → None. For each kind, render a 4 s 1280×720 `testsrc2` through `reframe_graph` with ffmpeg and assert the output is 1080×1920 with duration ≈4.0 ±0.1. Also test a two-scene graph (STACKED 0–2 s, FIT_BLUR 2–4 s) → 1080×1920, ≈4.0 s. Then:

```python
def test_track_expression_clamped_and_deadbanded():
    expr = track_x_expr([(0, 0.5), (1, 0.52), (2, 0.9)], in_w=1920, crop_w=607)
    assert "if(lt(t," in expr and evaluate(expr, t=0.5) == pytest.approx(evaluate(expr, t=0)) # deadband
    assert 0 <= evaluate(expr, t=2) <= 1920 - 607
```

(`track_x_expr` is a public helper in reframe.py; `evaluate` is a tiny test helper that substitutes `t` and evaluates the if/lt/lerp expression in Python.)

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS, plus the full suite and ruff.
- [ ] **Step 5: Commit.** Message: `feat: ffmpeg reframe graphs for STACKED/FACE_TRACK/SPLIT/FIT_BLUR`

---

### Task 9: ASS captions + hook title

**Files:**
- Create: `apps/server/hookline/captions/__init__.py`, `apps/server/hookline/captions/ass.py`, `apps/server/hookline/media/subtitles.py`, `assets/fonts/Anton-Regular.ttf`, `assets/fonts/OFL.txt` (from https://github.com/google/fonts/tree/main/ofl/anton)
- Test: `apps/server/tests/test_captions.py`

**Interfaces:**
- Consumes: remapped `Word`s (Task 3)
- Produces:
  - `group_words(words: list[Word], max_words: int = 3, max_chars: int = 18, max_gap: float = 0.5) -> list[list[Word]]`: a group breaks after sentence punctuation, after a gap longer than `max_gap`, or at the word or character limits.
  - `build_ass(words: list[Word], hook_text: str, duration: float, caption_y: int = 1300) -> str`:
    - Header: `PlayResX: 1080`, `PlayResY: 1920`, `WrapStyle: 2`, `ScaledBorderAndShadow: yes`.
    - Style `Cap`: font `Anton`, size 96, white, black outline 8, alignment 5 (centre), with `{\pos(540,caption_y)}`.
    - Style `Hook`: font `Anton`, size 84, white on a black box (`BorderStyle: 3`), positioned at `{\pos(540,260)}`, shown from 0 to `min(3.0, duration)`.
    - One `Dialogue` per word, spanning `[word.start, next word in group's start or word.end]`. Each shows the whole group in uppercase, with the active word wrapped in `{\c&H00FFFF&}` … `{\r}`.
    - Escape `{` `}` `\` in word text (`\{`, `\}`, `\\`).
  - `caption_y_for(layouts) -> int`: 640 (the seam) when the first scene is STACKED, otherwise 1300.
  - `subtitles_filter(ass_path: Path, fonts_dir: Path) -> str`: `subtitles='<p>':fontsdir='<d>'`, with absolute forward-slash paths, the drive colon escaped as `\:`, and `'` escaped.
  - `fonts_dir() -> Path` (`repo_root() / "assets" / "fonts"`)

- [ ] **Step 1: Write the failing tests**

```python
def test_group_breaks_on_punctuation_and_gap():
    g = group_words([w(0,.3,"No"), w(.3,.6,"way."), w(.7,1,"I"), w(2,2.3,"swear")])
    assert [[x.text for x in grp] for grp in g] == [["No", "way."], ["I"], ["swear"]]

def test_ass_highlights_each_word_once():
    ass = build_ass([w(0,.3,"no"), w(.3,.6,"way")], hook_text="Wait for it", duration=5)
    events = [l for l in ass.splitlines() if l.startswith("Dialogue")]
    assert len(events) == 3 and "{\\c&H00FFFF&}NO{\\r} WAY" in events[1] and "NO {\\c&H00FFFF&}WAY" in events[2]

def test_ass_escapes_special_chars():
    ass = build_ass([w(0, .5, "{hi}\\")], hook_text="x", duration=1)
    assert "\\{HI\\}\\\\" in ass

@pytest.mark.integration
def test_burns_captions_in_emoji_space_dir(fixture_video, tmp_path):
    d = tmp_path / "my clips 🎮"; d.mkdir()
    ass = d / "c.ass"; ass.write_text(build_ass([w(0,.5,"héllo"), w(.5,1,"😱")], "Hook!", 2), encoding="utf-8")
    run ffmpeg -t 2 -i fixture_video -vf subtitles_filter(ass, fonts_dir()) -f null -   # exit 0
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS, plus the full suite and ruff.
- [ ] **Step 5: Commit.** Message: `feat: word-by-word ASS captions and hook title`

---

### Task 10: Per-clip render + pipeline wiring

**Files:**
- Create: `apps/server/hookline/pipeline/render.py`
- Delete: `apps/server/hookline/pipeline/cut.py` (update its tests to target `render_clip`)
- Modify: `apps/server/hookline/pipeline/run.py`, `apps/server/hookline/config.py`, `apps/server/hookline/cli.py`, `apps/server/hookline/doctor.py`
- Test: `apps/server/tests/test_render.py`, `apps/server/tests/test_pipeline.py`, `apps/server/tests/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - New `Settings` fields:
    - `vertical: bool = True`
    - `captions: bool = True`
    - `max_pause_s: float = 0.4`
    - `silence_db: float = -32.0`
    - `facecam: tuple[float, float, float, float] | None = None`
    - `chat: bool = True`
  - `render_clip(src: Path, wav: Path, pick: ClipPick, words: list[Word], work_dir: Path, dst: Path, settings: Settings, detector: FaceDetector | None) -> dict`, which writes `dst` and returns sidecar extras. Steps:
    1. `silences = detect_silences(wav, pick.segments()[0][0] - 1, pick.end + 1, settings.silence_db)`
    2. `pick = refine_boundaries(pick, words, silences)`
    3. `keep = keep_ranges(pick.segments(), silences, settings.max_pause_s)`
    4. Write the intermediate `work_dir/NN.mp4` with `cut_segments(src, keep, ..., encoder="libx264")`, overriding the quality to `-crf 12`. Add a `quality: list[str] | None = None` parameter to `cut_segments` that replaces `encoder_args`' rate control when given.
    5. `crop = detect_crop(intermediate)`, then `layouts = plan_layouts(intermediate, detector, settings.facecam)` if vertical, else none. Face coordinates are relative to the uncropped frame, so convert them into the cropped frame before building the graph. The horizontal path applies `crop` too.
    6. `cap_words = remap_words(words, keep)`, then `ass = build_ass(cap_words, pick.hook_text, duration, caption_y_for(layouts))`.
    7. Final encode: `-filter_complex reframe_graph(...)` with `[vout]subtitles=...` appended, and audio copied as AAC 192k. If not vertical, scale only. If no captions, skip the subtitles filter.
    8. Return `{"layout": [l.model_dump() for l in layouts], "keep_ranges": keep, "removed_s": original - kept}`.
  - run_job stages:
    1. ingest (+ `try_fetch_chat` when `settings.chat`)
    2. audio
    3. transcribe
    4. signals: `signals.json` cached, holding `loudness = rms_per_second(wav)` and `chat_rate` if `chat.json` exists
    5. select: pass `markers=heat_markers(signals)`. picks.json now stores `{"key": {...}, "picks": [...]}`, where the key is `clips`, `select_model`, `min_clip_s` and `max_clip_s`. A key mismatch or the old list format means re-select (backlog item).
    6. render: each clip to `output/<job>/NN-<slug>.mp4` plus a sidecar with the extras
  - CLI flags: `--horizontal`, `--no-captions`, `--no-chat`, `--facecam X,Y,W,H` (normalized floats; validated, with a one-line error on bad input). The log prints each clip's layouts and how much dead air was removed.
  - Doctor: optional rows for the YuNet model, TwitchDownloaderCLI and the Anton font (required, since it is committed). The whisper.cpp row notes DTW.

- [ ] **Step 1: Write the failing tests.**
  - `test_render_clip_vertical_with_captions`: the fixture video, words from FakeTranscriber, and a fake detector returning no faces → output is 1080×1920, the sidecar has `layout[0]["kind"] == "FIT_BLUR"`, and the duration ≈ the sum of the keep ranges ±0.2.
  - `test_render_clip_horizontal_no_captions`: the output keeps the source aspect ratio.
  - `test_run_job_end_to_end`: update M1's test to assert 1080×1920 outputs and that signals.json is cached.
  - `test_picks_cache_key_mismatch_reselects`.
  - `test_cli_bad_facecam_exits_1`.
- [ ] **Step 2:** Run → FAIL. **Step 3:** Implement. **Step 4:** Run → PASS, plus the full suite and ruff.
- [ ] **Step 5: Commit.** Message: `feat: vertical captioned clip render wired into the pipeline`

---

### Task 11: Real run + docs

**Files:**
- Modify: `README.md` (Status → M2; features: vertical layouts, captions, dead-air removal, chat-aware picks; flags), `docs/superpowers/m2-backlog.md` (strike the done items, add new findings)

- [ ] **Step 1:** Delete `apps/server/workspace/e87398ed46/transcript.json` and `picks.json`, then `uv run hookline run "https://www.youtube.com/watch?v=9mrYGaeeg04"`. Record per-stage timings.
- [ ] **Step 2: Verify against the audio, not the transcript.**
  1. For each output clip, run `silencedetect=noise=-32dB:d=0.1` on its first and last 0.3 s, and report whether each clip starts and ends at or near silence.
  2. ffprobe each clip for 1080×1920.
  3. Report the removed dead-air seconds per clip and each clip's layout kinds.
  4. Extract 3 frames per clip (`-ss 1`, mid, end−1) to `workspace/<job>/review/` as PNGs and look at them: captions legible, hook title visible in the first 3 s, face framed.
- [ ] **Step 3:** If the maintainer provides a Twitch VOD URL, run it too and report the chat-marker count and the layouts chosen. Otherwise note that this check is pending.
- [ ] **Step 4:** Update the README and backlog, then commit with the message `docs: M2 quickstart and backlog`.
