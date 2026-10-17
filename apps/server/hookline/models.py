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


class ClipPick(BaseModel):
    start: float
    end: float
    title: str
    hook_text: str
    reason: str
    scores: RubricScores
    cold_open: tuple[float, float] | None = None
    edit_beats: list[str] = []

    @property
    def duration(self) -> float:
        return sum(e - s for s, e in self.segments())

    def segments(self) -> list[tuple[float, float]]:
        main = (self.start, self.end)
        return [self.cold_open, main] if self.cold_open else [main]
