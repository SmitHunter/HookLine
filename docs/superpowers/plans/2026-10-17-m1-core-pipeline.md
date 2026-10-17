# M1 Core Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `hookline run <file|url>` turns a long video into 5 raw (un-reframed) clips, each with Claude's hook text, scores and reason, from the command line.

**Architecture:** Python package `hookline` in `apps/server/`, managed by uv. The pipeline is a sequence of stages (ingest → transcribe → select → cut). Each stage writes an artifact into `workspace/<job_id>/`, and a stage is skipped when its artifact already exists. Transcription and the Claude call sit behind small Protocols, so tests inject fakes and the real backends are swappable.

**Tech Stack:** Python 3.12 (uv), typer, pydantic v2, yt-dlp, faster-whisper, anthropic SDK, ffmpeg (subprocess), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-10-17-hookline-design.md`

**Out of scope for M1** (later milestones in the spec): Twitch chat download and heat signals, `cropdetect`, dead-air removal, reframing, captions, web UI, renderers.

## Global Constraints

- Python `>=3.12,<3.13`, managed with `uv` (project at `apps/server/`). Format and lint with `ruff`.
- Repo licence: MIT. Never copy code from SupoClip (AGPL).
- Every subprocess call uses list args, never `shell=True`. All paths use `pathlib`, because paths with spaces are common on Windows.
- Default selection model `claude-opus-5-5`; default LLM backend `claude-code` (the `claude` CLI). API backend reads `ANTHROPIC_API_KEY`.
- Clip length target 20–60s (`min_clip_s=20`, `max_clip_s=60`); 5 clips by default; inputs shorter than 120s skip selection ("polish mode").
- Transcriber default `auto`: whisper.cpp if a binary and model are configured, otherwise faster-whisper (CUDA float16 if available, else CPU int8). Model `large-v3-turbo`.
- Encoder preference: `h264_nvenc` → `h264_amf` → `libx264`, each confirmed with a 1-frame probe encode.
- Artifacts are written atomically (write to `*.tmp`, then `os.replace`).
- The dev machine has an AMD RX 7800 XT (no CUDA), ffmpeg 8.1 (gyan full build), Node 24, the claude CLI and VS Build Tools 2022, but no CMake and no Vulkan SDK.

## Review Focus

1. **Paths with spaces or non-ASCII characters** (source files like `D:\My Videos\stream 🎮.mp4`, Twitch titles with emoji) → everything works and output filenames are ASCII slugs. Pinned in Task 4 (cut) and Task 6 (pipeline).
2. **Claude returns impossible timestamps** (`end <= start`, `start` beyond the video, overlapping picks) → the bad pick is dropped or clamped and the job never crashes. Pinned in Task 5.
3. **Claude returns invalid JSON or too few valid picks** → retry once, then continue with the valid picks and log a warning. Pinned in Task 5.
4. **Transcript with no punctuation** (casual stream speech often comes back unpunctuated) → sentence splitting falls back to pauses, so snapping still lands on natural boundaries. Pinned in Task 2.
5. **Job killed mid-stage, then re-run** → finished stages are reused and a half-written artifact is never treated as finished. Pinned in Task 6.

---

## File structure

```
hookline/
├─ LICENSE                       MIT
├─ README.md                     stub: what it is + M1 quickstart (full README is M6)
├─ .gitignore                    workspace/, output/, tools/bin/, .venv, __pycache__, *.tmp
├─ scripts/build-whisper-vulkan.ps1
└─ apps/server/
   ├─ pyproject.toml             package "hookline", console script `hookline = hookline.cli:app`
   ├─ hookline/
   │  ├─ config.py               Settings
   │  ├─ fsutil.py               atomic_write_text, slugify
   │  ├─ media/ffmpeg.py         probe_duration, extract_audio, pick_encoder, cut_segments
   │  ├─ models.py               Word, Segment, Transcript, Sentence, RubricScores, ClipPick
   │  ├─ transcript_text.py      split_sentences, compact_transcript
   │  ├─ transcribe/             base.py (Transcriber), faster_whisper.py, whisper_cpp.py, __init__.py (get_transcriber)
   │  ├─ llm/                    base.py (LLMBackend, LLMError), claude_code.py, anthropic_api.py, __init__.py (make_llm)
   │  ├─ pipeline/               ingest.py, select.py, cut.py, run.py
   │  ├─ doctor.py
   │  └─ cli.py
   └─ tests/
      ├─ conftest.py             fixture video generator, FakeLLM, FakeTranscriber, sample transcript
      ├─ fixtures/whisper_cpp_sample.json
      └─ test_*.py               one per module
```

---

### Task 1: Repo scaffold + ffmpeg helpers

**Files:**
- Create: `LICENSE`, `README.md` (stub), `.gitignore`, `apps/server/pyproject.toml`, `apps/server/hookline/__init__.py`, `apps/server/hookline/config.py`, `apps/server/hookline/fsutil.py`, `apps/server/hookline/media/__init__.py`, `apps/server/hookline/media/ffmpeg.py`
- Test: `apps/server/tests/conftest.py`, `apps/server/tests/test_ffmpeg.py`, `apps/server/tests/test_fsutil.py`

**Interfaces:**
- Produces:
  - `Settings` (pydantic `BaseModel`; `Settings.from_env()` reads `HOOKLINE_<FIELD>` env vars). Fields: `workspace_dir: Path = Path("workspace")`, `output_dir: Path = Path("output")`, `llm_backend: Literal["claude-code","api"] = "claude-code"`, `select_model: str = "claude-opus-5-5"`, `transcriber: Literal["auto","faster-whisper","whisper-cpp"] = "auto"`, `whisper_model: str = "large-v3-turbo"`, `whisper_cpp_bin: Path | None = None`, `whisper_cpp_model: Path | None = None`, `clips: int = 5`, `min_clip_s: float = 20`, `max_clip_s: float = 60`, `short_input_s: float = 120`, `max_prompt_tokens: int = 600_000`, `cookies_from_browser: str | None = None`
  - `atomic_write_text(path: Path, text: str) -> None`; `slugify(text: str, max_len: int = 40) -> str` (ASCII, lowercase, hyphens, never empty: falls back to `"clip"`)
  - `probe_duration(path: Path) -> float`
  - `extract_audio(src: Path, dst: Path) -> Path` (16 kHz mono PCM WAV)
  - `pick_encoder() -> str` (cached with `functools.cache`)
  - `cut_segments(src: Path, segments: list[tuple[float, float]], dst: Path, encoder: str) -> Path`
  - `FFmpegError(Exception)` carries the last 20 lines of stderr
- Test fixture: `fixture_video(tmp_path_factory) -> Path`, session-scoped. Generates a 150s, 320×180, 25fps `testsrc2` video with a `sine` audio track via ffmpeg lavfi, into a directory whose name contains a space and an emoji (`"my clips 🎮"`).

- [ ] **Step 1: Write the failing tests**

```python
def test_probe_duration(fixture_video):
    assert probe_duration(fixture_video) == pytest.approx(150.0, abs=0.1)

def test_extract_audio_is_16k_mono(fixture_video, tmp_path):
    wav = extract_audio(fixture_video, tmp_path / "a.wav")
    info = ffprobe_stream(wav)  # test helper using ffprobe -show_streams -of json
    assert info["sample_rate"] == "16000" and info["channels"] == 1

def test_pick_encoder_returns_known():
    assert pick_encoder() in {"h264_nvenc", "h264_amf", "libx264"}

def test_cut_segments_concatenates(fixture_video, tmp_path):
    out = cut_segments(fixture_video, [(100.0, 102.0), (10.0, 20.0)], tmp_path / "c.mp4", "libx264")
    assert probe_duration(out) == pytest.approx(12.0, abs=0.15)

def test_cut_segments_raises_ffmpeg_error_on_bad_input(tmp_path):
    with pytest.raises(FFmpegError):
        cut_segments(tmp_path / "missing.mp4", [(0, 1)], tmp_path / "x.mp4", "libx264")

def test_slugify():
    assert slugify("She said WHAT?! 😱") == "she-said-what"
    assert slugify("😱😱") == "clip"

def test_atomic_write_leaves_no_tmp(tmp_path):
    atomic_write_text(tmp_path / "a.json", "{}")
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd apps/server && uv run pytest -q`
Expected: FAIL / errors on import (`hookline.media.ffmpeg` missing)

- [ ] **Step 3: Implement**

Create the scaffold files above. `pyproject.toml` declares deps `typer, pydantic>=2, yt-dlp, faster-whisper, anthropic, rich` and dev deps `pytest, ruff`, `requires-python = ">=3.12,<3.13"`, plus the console script. `cut_segments` opens one input per segment (`-ss s -to e -i src`, so seeking stays fast on multi-hour VODs), joins them with the `concat` filter (`n=len(segments):v=1:a=1`), encodes with `-c:a aac -b:a 192k -movflags +faststart`, and writes to `dst.with_suffix(".tmp.mp4")` before `os.replace` to `dst`. `pick_encoder` parses `ffmpeg -hide_banner -encoders`, then makes a 1-frame `-f null -` probe encode for each candidate in preference order.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd apps/server && uv sync && uv run pytest -q && uv run ruff check .`
Expected: all pass, ruff clean

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: scaffold hookline server package with ffmpeg helpers"
```

---

### Task 2: Transcript models + sentence splitting + compaction

**Files:**
- Create: `apps/server/hookline/models.py`, `apps/server/hookline/transcript_text.py`
- Test: `apps/server/tests/test_transcript_text.py`, plus a `sample_transcript` fixture in `conftest.py`

**Interfaces:**
- Produces (pydantic models, all times are seconds as `float`):
  - `Word(start, end, text: str, prob: float = 1.0)`
  - `Segment(start, end, text: str, words: list[Word])`
  - `Transcript(language: str, duration: float, segments: list[Segment])` with `.words() -> list[Word]`
  - `Sentence(start, end, text: str)`
  - `RubricScores(hook: int, context: int, payoff: int, emotion: int, length_fit: int, pacing: int, loop: int, safety: int)` with `.total -> int` (sum; the per-criterion maxima are 25/15/15/15/10/10/5/5)
  - `ClipPick(start: float, end: float, cold_open: tuple[float, float] | None = None, title: str, hook_text: str, reason: str, scores: RubricScores, edit_beats: list[str] = [])` with `.duration -> float` and `.segments() -> list[tuple[float, float]]` (cold_open first when present, then `(start, end)`)
  - `split_sentences(words: list[Word], max_gap: float = 0.7, max_len_s: float = 25.0) -> list[Sentence]`
  - `compact_transcript(sentences: list[Sentence]) -> str` gives one line per sentence: `[mm:ss.s] text`, using `[h:mm:ss.s]` once the time passes an hour

- [ ] **Step 1: Write the failing tests**

```python
def w(s, e, t): return Word(start=s, end=e, text=t)

def test_split_on_punctuation():
    words = [w(0,.4,"Hi."), w(.5,.9,"How"), w(1,1.3,"are"), w(1.4,1.8,"you?")]
    assert [s.text for s in split_sentences(words)] == ["Hi.", "How are you?"]

def test_split_falls_back_to_pauses_without_punctuation():
    words = [w(0,.4,"no"), w(.5,.9,"punctuation"), w(2.0,2.4,"here"), w(2.5,2.9,"either")]
    sents = split_sentences(words)
    assert [s.text for s in sents] == ["no punctuation", "here either"]
    assert (sents[1].start, sents[1].end) == (2.0, 2.9)

def test_split_caps_runaway_sentences():
    words = [w(i, i + .5, "yap") for i in range(60)]  # 60s, no punctuation, gaps 0.5
    assert all(s.end - s.start <= 25.0 for s in split_sentences(words))

def test_compact_format_and_hours():
    out = compact_transcript([Sentence(start=65.25, end=70, text="Hello there."),
                              Sentence(start=3725.0, end=3726, text="Late.")])
    assert out.splitlines() == ["[01:05.2] Hello there.", "[1:02:05.0] Late."]

def test_clip_pick_segments_put_cold_open_first(sample_pick):
    p = sample_pick.model_copy(update={"cold_open": (50.0, 52.5)})
    assert p.segments() == [(50.0, 52.5), (30.0, 55.0)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_transcript_text.py -q` → FAIL (import error)

- [ ] **Step 3: Implement** the models and the two functions in the files above. A sentence ends after a word whose text ends in `.`, `?` or `!` (trailing quotes and brackets stripped first), when the gap to the next word exceeds `max_gap`, or when adding the next word would push it past `max_len_s`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -q && uv run ruff check .` → PASS

- [ ] **Step 5: Commit** — `git commit -am "feat: transcript models, sentence splitting, compaction"` (after `git add` of new files)

---

### Task 3: Transcription backends

**Files:**
- Create: `apps/server/hookline/transcribe/__init__.py`, `base.py`, `faster_whisper.py`, `whisper_cpp.py`
- Test: `apps/server/tests/test_transcribe.py`, `apps/server/tests/fixtures/whisper_cpp_sample.json`

**Interfaces:**
- Consumes: `Transcript`, `Segment`, `Word` (Task 2); `extract_audio` (Task 1); `Settings`
- Produces:
  - `class Transcriber(Protocol): def transcribe(self, wav: Path, on_progress: Callable[[float], None] | None = None) -> Transcript`
  - `FasterWhisperBackend(model: str, device: str, compute_type: str)`. Uses `word_timestamps=True` and `vad_filter=True`, and reports progress as `segment.end / info.duration`.
  - `WhisperCppBackend(bin: Path, model: Path)`. Runs `bin -m model -f wav -ojf -of <tmp stem> -ml 1 -sow -pp` and parses the result with `parse_whisper_cpp_json(data: dict, duration: float) -> Transcript`.
  - `parse_whisper_cpp_json(data: dict, duration: float) -> Transcript`. The input holds whisper.cpp `-ojf` output, where `-ml 1 -sow` produces one word per entry in `transcription[]`. Each entry's `offsets.from/to` is in ms and its `text` has a leading space. Words are regrouped into `Segment`s with `split_sentences` boundaries, and words whose text is empty after stripping are dropped.
  - `get_transcriber(settings: Settings) -> Transcriber`. Uses `whisper-cpp` when explicitly selected, or when `auto` is set and both `whisper_cpp_bin` and `whisper_cpp_model` exist. Otherwise uses faster-whisper: CUDA float16 if `ctranslate2.get_cuda_device_count() > 0`, else CPU int8.

- [ ] **Step 1: Write the failing tests**

```python
def test_parse_whisper_cpp_json():
    data = json.loads((FIXTURES / "whisper_cpp_sample.json").read_text(encoding="utf-8"))
    t = parse_whisper_cpp_json(data, duration=5.0)
    words = t.words()
    assert [x.text for x in words][:3] == ["Okay", "chat,", "watch"]
    assert words[0].start == pytest.approx(0.32) and words[0].end == pytest.approx(0.61)
    assert all(x.text.strip() for x in words)

def test_get_transcriber_auto_falls_back_to_faster_whisper(tmp_path):
    s = Settings(whisper_cpp_bin=tmp_path / "nope.exe", whisper_cpp_model=tmp_path / "nope.bin")
    assert isinstance(get_transcriber(s), FasterWhisperBackend)

def test_get_transcriber_auto_prefers_whisper_cpp_when_present(tmp_path):
    (tmp_path / "w.exe").write_bytes(b""); (tmp_path / "m.bin").write_bytes(b"")
    s = Settings(whisper_cpp_bin=tmp_path / "w.exe", whisper_cpp_model=tmp_path / "m.bin")
    assert isinstance(get_transcriber(s), WhisperCppBackend)

@pytest.mark.integration
def test_faster_whisper_tiny_on_speech(speech_wav):
    t = FasterWhisperBackend("tiny.en", "cpu", "int8").transcribe(speech_wav)
    assert "hookline" in " ".join(w.text.lower() for w in t.words())
```

The fixture JSON is hand-written in the `-ojf` shape: `{"transcription":[{"offsets":{"from":320,"to":610},"text":" Okay", ...}, ...]}`, with about 8 words and one empty-text entry. `speech_wav` is a session fixture that generates speech with Windows SAPI (`powershell -Command Add-Type -AssemblyName System.Speech; ... SetOutputToWaveFile`) saying "Welcome to hookline. This is a test." and skips on non-Windows. Register the `integration` marker in `pyproject.toml`.

- [ ] **Step 2: Run** `uv run pytest tests/test_transcribe.py -q -m "not integration"` → FAIL
- [ ] **Step 3: Implement** the files above.
- [ ] **Step 4: Run** `uv run pytest -q` (integration included; the first run downloads the tiny model) → PASS
- [ ] **Step 5: Commit** — `feat: faster-whisper and whisper.cpp transcription backends`

---

### Task 4: Claude backends

**Files:**
- Create: `apps/server/hookline/llm/__init__.py`, `base.py`, `claude_code.py`, `anthropic_api.py`
- Test: `apps/server/tests/test_llm.py`

**Interfaces:**
- Produces:
  - `class LLMBackend(Protocol): def complete_json(self, system: str, prompt: str, schema: dict, model: str) -> dict`
  - `class LLMError(Exception)`
  - `ClaudeCodeBackend(runner: Callable[..., subprocess.CompletedProcess] = subprocess.run, timeout_s: int = 900)`
    - Command: `["claude", "-p", "--output-format", "json", "--json-schema", json.dumps(schema), "--model", model, "--system-prompt", system, "--tools", ""]`, with `prompt` sent on **stdin** (transcripts are too long for a Windows command line), `encoding="utf-8"`, and `cwd` set to a temp dir so project CLAUDE.md files don't leak in.
    - Returns `payload["structured_output"]`.
    - Raises `LLMError` on non-zero exit, `is_error: true`, or a missing `structured_output`.
  - `AnthropicApiBackend(client: anthropic.Anthropic | None = None, effort: str = "high")`
    - Request: `client.beta.messages.stream(model=model, max_tokens=64000, system=system, messages=[{"role":"user","content":prompt}], output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}}, betas=["server-side-fallback-2026-07-01"], fallbacks="default")`, then `.get_final_message()`.
    - Raises `LLMError` when `stop_reason` is `"refusal"` or `"max_tokens"`; otherwise returns `json.loads` of the first text block.
    - Server-side refusal fallback is on by default. The README mentions it.
  - `make_llm(settings: Settings) -> LLMBackend`

- [ ] **Step 1: Write the failing tests**

```python
def fake_run(stdout, code=0):
    calls = []
    def run(args, **kw):
        calls.append((args, kw)); return subprocess.CompletedProcess(args, code, stdout, "")
    return run, calls

def test_claude_code_sends_prompt_on_stdin_and_returns_structured_output():
    run, calls = fake_run(json.dumps({"is_error": False, "structured_output": {"clips": []}}))
    out = ClaudeCodeBackend(runner=run).complete_json("sys", "BIG PROMPT", {"type": "object"}, "claude-opus-5-5")
    args, kw = calls[0]
    assert out == {"clips": []}
    assert kw["input"] == "BIG PROMPT" and "BIG PROMPT" not in args
    assert args[args.index("--model") + 1] == "claude-opus-5-5"

@pytest.mark.parametrize("stdout,code", [("not json", 0), (json.dumps({"is_error": True, "result": "x"}), 0), ("{}", 1)])
def test_claude_code_errors(stdout, code):
    run, _ = fake_run(stdout, code)
    with pytest.raises(LLMError):
        ClaudeCodeBackend(runner=run).complete_json("s", "p", {}, "m")

def test_api_backend_refusal_raises(fake_anthropic_client_factory):
    client = fake_anthropic_client_factory(stop_reason="refusal", text="")
    with pytest.raises(LLMError):
        AnthropicApiBackend(client=client).complete_json("s", "p", {}, "claude-opus-5-5")

def test_api_backend_parses_text(fake_anthropic_client_factory):
    client = fake_anthropic_client_factory(stop_reason="end_turn", text='{"clips": [1]}')
    assert AnthropicApiBackend(client=client).complete_json("s", "p", {}, "m") == {"clips": [1]}
```

`fake_anthropic_client_factory` (in conftest) builds a minimal object where `client.beta.messages.stream(**kw)` is a context manager whose `.get_final_message()` returns a `SimpleNamespace(stop_reason=..., content=[SimpleNamespace(type="text", text=...)])`, and it records `kw` so a test can assert `output_config["format"]["type"] == "json_schema"`.

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** `uv run pytest -q` → PASS
- [ ] **Step 5: Smoke test (real CLI, about $0.01):** `uv run python -c "from hookline.llm import ClaudeCodeBackend as B; print(B().complete_json('Answer as JSON.', 'Pick 7.', {'type':'object','properties':{'n':{'type':'integer'}},'required':['n']}, 'claude-haiku-5-5'))"` → `{'n': 7}`
- [ ] **Step 6: Commit** — `feat: Claude Code CLI and Anthropic API backends`

---

### Task 5: Clip selection

**Files:**
- Create: `apps/server/hookline/pipeline/__init__.py`, `apps/server/hookline/pipeline/select.py`
- Test: `apps/server/tests/test_select.py` (`FakeLLM` lives in conftest; it returns queued dicts or raises queued exceptions, and records its calls)

**Interfaces:**
- Consumes: `LLMBackend`, `LLMError` (Task 4); `Transcript`, `Sentence`, `ClipPick`, `RubricScores`, `split_sentences`, `compact_transcript` (Task 2); `Settings`
- Produces:
  - `SYSTEM_PROMPT: str`. The rubric from the spec, verbatim: Hook 25 · Standalone context 15 · Payoff 15 · Emotional intensity 15 · Length fit 10 · Pacing 10 · Loop potential 5 · Platform safety 5. It also contains:
    - the 20–60s target ("20–45s usually wins; 60s only if tension holds")
    - start and end on complete sentences
    - the hook within the first 3 seconds; if the natural opening is weak, set `cold_open` to a ≤3s line from inside the clip
    - `hook_text` is on-screen text of ≤8 words
    - candidates are ranked best first
  - `SELECTION_SCHEMA: dict`. JSON Schema `{"clips": [ {start, end, cold_open_start|null, cold_open_end|null, title, hook_text, reason, scores{8 ints}, edit_beats[str]} ]}`. It uses `additionalProperties: false` and `required` everywhere, and no numeric min/max keywords (range checks happen in Python).
  - `build_prompt(compact: str, n: int, duration: float, exclude: list[tuple[float, float]]) -> str`
  - `snap_to_sentences(pick: ClipPick, sentences: list[Sentence], duration: float) -> ClipPick`
    - `start` becomes the latest sentence start ≤ `pick.start + 0.5` (the first sentence if there is none).
    - `end` becomes the earliest sentence end ≥ `pick.end - 0.5` (the last sentence if there is none).
    - Both are clamped to `[0, duration]`.
  - `dedupe_overlaps(picks: list[ClipPick], max_overlap: float = 0.5) -> list[ClipPick]`. Walks picks by `scores.total` descending and keeps one only if `overlap / min(len_a, len_b) <= max_overlap` against every pick already kept.
  - `select_clips(transcript: Transcript, llm: LLMBackend, settings: Settings, exclude: list[tuple[float, float]] = []) -> list[ClipPick]`
    1. If `transcript.duration < settings.short_input_s`, return `[polish_pick(transcript)]` with no LLM call.
    2. Ask for `settings.clips + 3` candidates.
    3. Parse each candidate into a `ClipPick`. Drop it when `end <= start`, when `start >= duration`, or when the duration after snapping falls outside `[min_clip_s * 0.75, max_clip_s * 1.25]`. Drop a `cold_open` that isn't inside `[start, end]` or is longer than 3.5s.
    4. Snap, dedupe, and return the top `settings.clips` by total score, sorted by `start`.
    5. On `LLMError` or `pydantic.ValidationError` of the whole payload, retry once. If the second attempt also fails, raise `LLMError`. If fewer than `clips` remain valid, log a warning and return what's left.
    6. When `len(compact) / 4 > settings.max_prompt_tokens`, split into 120-minute windows with a 2-minute overlap, request `settings.clips` candidates per window, then merge them, dedupe and take the top N.
  - `polish_pick(transcript: Transcript) -> ClipPick`. Covers the whole input. `hook_text` is the first sentence truncated to 8 words, and `scores` are all zero.

- [ ] **Step 1: Write the failing tests**

```python
def test_select_returns_top_n_sorted_by_start(long_transcript, fake_llm, settings):
    fake_llm.queue(make_payload(8))  # 8 valid non-overlapping candidates, descending scores
    picks = select_clips(long_transcript, fake_llm, settings)
    assert len(picks) == 5 and [p.start for p in picks] == sorted(p.start for p in picks)
    assert fake_llm.calls[0]["model"] == "claude-opus-5-5"

def test_snap_moves_to_sentence_bounds(long_sentences):
    p = snap_to_sentences(pick(start=31.2, end=54.1), long_sentences, duration=600)
    assert p.start in {s.start for s in long_sentences} and p.end in {s.end for s in long_sentences}

def test_impossible_timestamps_are_dropped(long_transcript, fake_llm, settings):
    fake_llm.queue(make_payload(6, extra=[raw(start=50, end=40), raw(start=9999, end=10020)]))
    picks = select_clips(long_transcript, fake_llm, settings)
    assert all(p.end > p.start and p.end <= long_transcript.duration for p in picks)

def test_overlaps_deduped_keeping_higher_score():
    a, b = pick(0, 40, total=90), pick(10, 45, total=70)
    assert dedupe_overlaps([b, a]) == [a]

def test_retries_once_then_raises(long_transcript, fake_llm, settings):
    fake_llm.queue(LLMError("boom"), {"clips": "garbage"})
    with pytest.raises(LLMError):
        select_clips(long_transcript, fake_llm, settings)
    assert len(fake_llm.calls) == 2

def test_retry_recovers(long_transcript, fake_llm, settings):
    fake_llm.queue(LLMError("boom"), make_payload(8))
    assert len(select_clips(long_transcript, fake_llm, settings)) == 5

def test_short_input_skips_llm(short_transcript, fake_llm, settings):
    assert len(select_clips(short_transcript, fake_llm, settings)) == 1 and fake_llm.calls == []

def test_chunking_when_over_budget(long_transcript, fake_llm, settings):
    s = settings.model_copy(update={"max_prompt_tokens": 500})
    fake_llm.queue_forever(make_payload(5))
    select_clips(long_transcript, fake_llm, s)
    assert len(fake_llm.calls) > 1

def test_bad_cold_open_is_removed(long_transcript, fake_llm, settings):
    fake_llm.queue(make_payload(8, cold_open=(0.0, 10.0)))  # outside clip / too long
    assert all(p.cold_open is None for p in select_clips(long_transcript, fake_llm, settings))
```

`long_transcript` is a synthetic 600s transcript: 4-word sentences ending in "." every 3s. `make_payload`, `raw` and `pick` are test helpers in `test_select.py`.

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** `uv run pytest -q` → PASS
- [ ] **Step 5: Commit** — `feat: Claude clip selection with snapping, dedupe, retry, chunking`

---

### Task 6: Ingest, cut, pipeline runner, CLI, doctor

**Files:**
- Create: `apps/server/hookline/pipeline/ingest.py`, `cut.py`, `run.py`, `apps/server/hookline/doctor.py`, `apps/server/hookline/cli.py`
- Test: `apps/server/tests/test_pipeline.py`, `test_ingest.py`, `test_doctor.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `job_id_for(source: str) -> str`. The first 10 hex characters of sha1 over either the URL, or the absolute path plus size plus mtime for a local file.
  - `ingest(source: str, job_dir: Path, settings: Settings) -> Path`. Returns `job_dir / "source.<ext>"`.
    - Local files are hard-linked when on the same volume, otherwise copied.
    - URLs go through the `yt_dlp.YoutubeDL` Python API with `outtmpl=job_dir/"source.%(ext)s"`, format `bv*[height<=1080]+ba/b`, `merge_output_format="mp4"`, and `cookiesfrombrowser` when configured.
    - Skipped if `source.*` already exists.
  - `cut_clips(src: Path, picks: list[ClipPick], out_dir: Path, encoder: str) -> list[Path]`. Writes `NN-<slug(title)>.mp4` and a sibling `.json` (the `ClipPick` dump plus `"post_caption": hook_text`).
  - `run_job(source: str, settings: Settings, llm: LLMBackend, transcriber: Transcriber, log: Callable[[str], None] = print) -> list[Path]`
    - Stages, each skipped when its artifact exists: `source.*`, `audio.wav`, `transcript.json`, `picks.json`.
    - JSON artifacts are written with `atomic_write_text`.
    - Clips are written to `settings.output_dir / job_id`.
  - CLI (typer `app`):
    - `hookline run SOURCE [--clips N] [--backend claude-code|api] [--model M] [--out DIR]`
    - `hookline doctor` prints a rich table and exits 1 if any required check fails.
  - `run_doctor(which=shutil.which, run=subprocess.run) -> list[Check]`, with `Check(name: str, ok: bool, required: bool, detail: str)`.
    - Required: ffmpeg on PATH, ffmpeg has libass, at least one LLM backend (the `claude` CLI on PATH or `ANTHROPIC_API_KEY` set), yt_dlp importable.
    - Optional: h264_amf/h264_nvenc, whisper.cpp binary + model configured, CUDA available.

- [ ] **Step 1: Write the failing tests**

```python
def test_run_job_end_to_end_with_fakes(fixture_video, tmp_path, fake_llm, settings_tmp):
    fake_llm.queue(make_payload_for(duration=150, n=2, length=22))
    s = settings_tmp.model_copy(update={"clips": 2})
    clips = run_job(str(fixture_video), s, fake_llm, FakeTranscriber(duration=150))
    assert len(clips) == 2 and all(c.exists() and c.name.isascii() for c in clips)
    assert all(probe_duration(c) == pytest.approx(22, abs=0.6) for c in clips)
    assert all(c.with_suffix(".json").exists() for c in clips)

def test_rerun_uses_cache(fixture_video, fake_llm, settings_tmp):
    fake_llm.queue(make_payload_for(150, 2, 22))
    s = settings_tmp.model_copy(update={"clips": 2})
    t = FakeTranscriber(duration=150)
    run_job(str(fixture_video), s, fake_llm, t)
    run_job(str(fixture_video), s, fake_llm, t)
    assert t.calls == 1 and len(fake_llm.calls) == 1

def test_partial_artifact_is_not_trusted(fixture_video, fake_llm, settings_tmp):
    job = settings_tmp.workspace_dir / job_id_for(str(fixture_video)); job.mkdir(parents=True)
    (job / "transcript.json.tmp").write_text("{trunc", encoding="utf-8")
    fake_llm.queue(make_payload_for(150, 2, 22))
    t = FakeTranscriber(duration=150)
    run_job(str(fixture_video), settings_tmp.model_copy(update={"clips": 2}), fake_llm, t)
    assert t.calls == 1

def test_job_id_stable_for_url():
    assert job_id_for("https://www.twitch.tv/videos/1") == job_id_for("https://www.twitch.tv/videos/1")

def test_doctor_flags_missing_ffmpeg():
    checks = run_doctor(which=lambda name: None, run=fake_completed(""))
    assert any(c.name == "ffmpeg" and not c.ok and c.required for c in checks)
```

`FakeTranscriber(duration)` returns a deterministic transcript (sentences every 3s across the duration) and counts calls. `settings_tmp` points `workspace_dir` and `output_dir` at `tmp_path`. `fixture_video` lives in the space-and-emoji directory from Task 1.

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** `uv run pytest -q && uv run ruff check .` → PASS
- [ ] **Step 5: CLI smoke:** `uv run hookline doctor` → table printed. ffmpeg, libass, claude CLI and yt-dlp show OK; whisper.cpp shows as optional/missing.
- [ ] **Step 6: Commit** — `feat: ingest, cut, resumable pipeline runner, CLI and doctor`

---

### Task 7: whisper.cpp Vulkan build (AMD GPU transcription)

**Requires the maintainer's OK before running:** it installs CMake and the Vulkan SDK system-wide via winget.

**Files:**
- Create: `scripts/build-whisper-vulkan.ps1`
- Modify: `README.md` (an "AMD / Intel GPU on Windows" section)

**Interfaces:**
- Produces: `tools/bin/whisper-cli.exe` (plus its DLLs) and `tools/models/ggml-large-v3-turbo.bin`. When both exist, the script prints the two env var lines (`HOOKLINE_WHISPER_CPP_BIN`, `HOOKLINE_WHISPER_CPP_MODEL`) to set.

- [ ] **Step 1: Write the script.**
  1. Install `Kitware.CMake` and `KhronosGroup.VulkanSDK` via `winget install -e --id`, skipping each if already present.
  2. Clone `https://github.com/ggml-org/whisper.cpp` at tag `v1.9.5` into `tools/src/whisper.cpp`.
  3. Load the VS 2022 Build Tools environment via `vswhere` + `Launch-VsDevShell.ps1`.
  4. Run `cmake -B build -DGGML_VULKAN=1 -DCMAKE_BUILD_TYPE=Release`, then `cmake --build build --config Release -j`.
  5. Copy `build/bin/Release/*` to `tools/bin/`.
  6. Download `ggml-large-v3-turbo.bin` from `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin` to `tools/models/`.
- [ ] **Step 2: Run it** (`powershell -ExecutionPolicy Bypass -File scripts/build-whisper-vulkan.ps1`). Expected: the build log shows `ggml_vulkan: Found 1 Vulkan devices: AMD Radeon RX 7800 XT`, and `tools/bin/whisper-cli.exe` exists.
- [ ] **Step 3: Verify:** with the env vars set, `uv run pytest -q -m integration` passes, and `uv run hookline doctor` shows whisper.cpp OK. Run a 10-minute real file and record the wall-clock time in the README GPU table.
- [ ] **Step 4: Commit** — `feat: whisper.cpp Vulkan build script for AMD/Intel GPUs`

---

### Task 8: Real VOD run + M1 README

**Files:**
- Modify: `README.md` (M1 quickstart: install uv, `uv sync`, `hookline doctor`, `hookline run <url>`, the copyright note from the spec, a note on server-side refusal fallback for the API backend)

- [ ] **Step 1:** Get a Twitch VOD URL from the maintainer, or use a long CC-licensed podcast from YouTube. Run `uv run hookline run "<url>"`.
- [ ] **Step 2: Verify:** 5 MP4s plus JSON files in `output/<job>/`. Every clip starts and ends on a complete sentence and lasts 15–75s. Re-running finishes in under 30s (all stages cached).
- [ ] **Step 3:** Show the maintainer the 5 clips' titles, hooks and reasons. Note any selection-prompt changes for M2.
- [ ] **Step 4: Commit** — `docs: M1 quickstart`
