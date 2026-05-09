"""Database storage and persistence for MedEvalKit."""

import hashlib
from pathlib import Path
from typing import Optional

from sqlmodel import Session, SQLModel, create_engine, select

from .schemas import Evaluation, Question, Response, Run

# Default database path
DEFAULT_DB_PATH = Path.home() / ".medevalkit" / "medevalkit.db"


def get_engine(db_path: Optional[Path] = None):
    """Create and return SQLAlchemy engine for the database."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    
    # Ensure directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create engine with SQLite
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    
    # Create tables if they don't exist
    SQLModel.metadata.create_all(engine)
    
    return engine


def persist_run(session: Session, run: Run) -> None:
    """Persist a run to the database."""
    session.add(run)
    session.commit()


def persist_question(session: Session, question: Question) -> None:
    """Persist a question, handling duplicates by ID."""
    existing = session.get(Question, question.id)
    if existing is None:
        session.add(question)
        session.commit()


def persist_response(session: Session, response: Response) -> None:
    """Persist a response to the database."""
    session.add(response)
    session.commit()


def persist_evaluation(session: Session, evaluation: Evaluation) -> None:
    """Persist an evaluation to the database."""
    session.add(evaluation)
    session.commit()


def get_run(session: Session, run_id: str) -> Optional[Run]:
    """Get a run by ID."""
    return session.get(Run, run_id)


def get_responses_for_run(session: Session, run_id: str) -> list[Response]:
    """Get all responses for a run."""
    statement = select(Response).where(Response.run_id == run_id)
    return list(session.exec(statement))


def get_evaluations_for_response(session: Session, response_id: str) -> list[Evaluation]:
    """Get all evaluations for a response."""
    statement = select(Evaluation).where(Evaluation.response_id == response_id)
    return list(session.exec(statement))


def response_cache_key(question_id: str, responder_model: str, prompt_hash: str) -> str:
    """Generate cache key for response lookup."""
    combined = f"{question_id}:{responder_model}:{prompt_hash}"
    return hashlib.sha256(combined.encode()).hexdigest()


def get_cached_response(
    session: Session, question_id: str, responder_model: str, prompt_hash: str
) -> Optional[Response]:
    """Look up a cached response by question and model."""
    # For simplicity, we'll search by question_id and parse_ok status
    # In production, you'd want a dedicated cache table with the hash
    statement = (
        select(Response)
        .where(Response.question_id == question_id)
        .where(Response.parse_ok == True)
        .order_by(Response.created_at.desc())
    )
    
    responses = list(session.exec(statement))
    
    # Filter by model (stored in related Run)
    for response in responses:
        run = session.get(Run, response.run_id)
        if run and run.responder_model == responder_model:
            return response
    
    return None