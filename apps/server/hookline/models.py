from typing import Literal

from pydantic import BaseModel, computed_field


class Word(BaseModel):
    start: float
    end: float
    text: str
    prob: float = 1.0


class Segment(BaseModel):
    start: float
    end: float
    text: str
    words: list[Word]


class Transcript(BaseModel):
    language: str
    duration: float
    segments: list[Segment]
    timing: Literal["segment", "dtw", "whisper"] = "segment"

    def words(self) -> list[Word]:
        return [w for seg in self.segments for w in seg.words]


class Sentence(BaseModel):
    start: float
    end: float
    text: str


class RubricScores(BaseModel):
    """Per-criterion maxima: 25/15/15/15/10/10/5/5."""

    hook: int
    context: int
    payoff: int
    emotion: int
    length_fit: int
    pacing: int
    loop: int
    safety: int

    @computed_field
    @property
    def total(self) -> int:
        return (
            self.hook
            + self.context
            + self.payoff
            + self.emotion
            + self.length_fit
            + self.pacing
            + self.loop
            + self.safety
        )


HookType = Literal[
    "bold_claim",
    "surprising_fact",
    "conflict",
    "confession",
    "question",
    "in_progress_action",
    "reaction_peak",
    "story_peak",
]
Verdict = Literal["post", "maybe", "skip"]
CommentRisk = Literal["low", "medium", "high"]


class ClipPick(BaseModel):
    start: float
    end: float
    title: str
    hook_text: str
    reason: str
    scores: RubricScores
    cold_open: tuple[float, float] | None = None
    edit_beats: list[str] = []
    # Selection v2. Defaults keep picks cached before these fields existed valid.
    first_words: str = ""
    last_words: str = ""
    hook_type: HookType | None = None
    verdict: Verdict = "maybe"
    post_caption: str = ""
    hashtags: list[str] = []
    search_title: str = ""
    category: str = ""
    comment_risk: CommentRisk | None = None

    @property
    def duration(self) -> float:
        return sum(e - s for s, e in self.segments())

    def segments(self) -> list[tuple[float, float]]:
        main = (self.start, self.end)
        return [self.cold_open, main] if self.cold_open else [main]
