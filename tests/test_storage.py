"""Tests for storage module."""

from datetime import datetime
from pathlib import Path

import pytest
from sqlmodel import Session

from medevalkit.schemas import Question, Run
from medevalkit.storage import get_engine, persist_question, persist_run


def test_engine_creation(tmp_path: Path):
    """Test database engine creation."""
    db_path = tmp_path / "test.db"
    engine = get_engine(db_path)
    
    assert db_path.exists()
    assert engine is not None


def test_persist_run(tmp_path: Path):
    """Test persisting a run."""
    engine = get_engine(tmp_path / "test.db")
    
    with Session(engine) as session:
        run = Run(
            id="test-run-123",
            dataset="medqa_50.jsonl",
            n_questions=5,
            responder_model="llama3.1:8b",
            responder_provider="ollama",
            judge_model="qwen2.5:7b-instruct",
            judge_provider="ollama",
            config={"temperature": 0.7, "seed": 42, "concurrency": 2},
            started_at=datetime.utcnow(),
        )
        persist_run(session, run)
        
        # Verify it was saved
        saved_run = session.get(Run, "test-run-123")
        assert saved_run is not None
        assert saved_run.dataset == "medqa_50.jsonl"
        assert saved_run.n_questions == 5


def test_persist_question_deduplication(tmp_path: Path):
    """Test that questions are deduplicated by ID."""
    engine = get_engine(tmp_path / "test.db")
    
    with Session(engine) as session:
        question = Question(
            id="q-123",
            source="test",
            text="What is the mechanism of action of aspirin?",
            ground_truth="Irreversible COX inhibition",
        )
        
        # Persist twice
        persist_question(session, question)
        persist_question(session, question)
        
        # Should still only have one
        all_questions = list(session.query(Question).all())
        assert len(all_questions) == 1