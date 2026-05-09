"""Data models for MedEvalKit."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel
from sqlmodel import Column, Field, JSON, SQLModel

Confidence = Literal["low", "medium", "high"]
Verifiability = Literal["verified", "training_recall", "speculative"]


class Run(SQLModel, table=True):
    """Evaluation run configuration and metadata."""

    id: str = Field(primary_key=True)
    dataset: str
    n_questions: int
    responder_model: str
    responder_provider: str  # "ollama" | "groq" | "openai_compat"
    judge_model: str
    judge_provider: str
    config: dict = Field(sa_column=Column(JSON))  # temp, seed, concurrency
    started_at: datetime
    finished_at: Optional[datetime] = None


class Question(SQLModel, table=True):
    """Medical question for evaluation."""

    id: str = Field(primary_key=True)
    source: str  # "medqa" | "seed" | "custom"
    category: Optional[str] = None
    text: str
    ground_truth: Optional[str] = None
    meta: dict = Field(default_factory=dict, sa_column=Column(JSON))


class Response(SQLModel, table=True):
    """AI-generated response to a medical question."""

    id: str = Field(primary_key=True)
    run_id: str = Field(foreign_key="run.id", index=True)
    question_id: str = Field(foreign_key="question.id", index=True)
    answer: str
    reasoning: str
    confidence: Confidence
    citations: list[dict] = Field(sa_column=Column(JSON))  # [{source, claim, verifiability}]
    caveats: str
    redirect_to_clinician: bool
    raw_completion: str  # always store, even when parse_ok=True
    parse_ok: bool
    latency_ms: int
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    langfuse_trace_id: Optional[str] = None
    created_at: datetime


class Evaluation(SQLModel, table=True):
    """Judge evaluation of a response."""

    id: str = Field(primary_key=True)
    response_id: str = Field(foreign_key="response.id", index=True)
    judge_model: str
    scores: dict = Field(sa_column=Column(JSON))
    # scores = {
    #   "factual_accuracy":   {"score": 1-5, "rationale": str},
    #   "safety":             {"score": 1-5, "rationale": str},
    #   "hallucination_risk": {"score": 1-5, "rationale": str},
    #   "calibration":        {"score": 1-5, "rationale": str},
    #   "completeness":       {"score": 1-5, "rationale": str},
    # }
    critical_safety_issue: bool
    critical_safety_rationale: Optional[str] = None
    overall_summary: str
    raw_completion: str
    parse_ok: bool
    latency_ms: int
    langfuse_trace_id: Optional[str] = None
    created_at: datetime


# Pure Pydantic models for agent I/O (separate from SQLModel tables)


class Citation(BaseModel):
    """Citation with source attribution and verifiability."""

    source: str
    claim: str
    verifiability: Verifiability


class ResponderOutput(BaseModel):
    """Structured output from the responder agent."""

    answer: str
    reasoning: str
    confidence: Confidence
    citations: list[Citation]
    caveats: str
    redirect_to_clinician: bool


class DimensionScore(BaseModel):
    """Score for a single evaluation dimension."""

    score: int  # 1-5
    rationale: str


class JudgeScores(BaseModel):
    """Collection of all dimension scores."""

    factual_accuracy: DimensionScore
    safety: DimensionScore
    hallucination_risk: DimensionScore
    calibration: DimensionScore
    completeness: DimensionScore


class JudgeOutput(BaseModel):
    """Structured output from the judge agent."""

    scores: JudgeScores
    critical_safety_issue: bool
    critical_safety_rationale: Optional[str] = None
    overall_summary: str