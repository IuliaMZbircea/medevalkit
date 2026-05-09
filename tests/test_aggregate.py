"""Tests for aggregation functions."""

from datetime import datetime

import pytest
from sqlmodel import Session

from medevalkit.aggregate import compute_aggregates
from medevalkit.schemas import Evaluation, Question, Response, Run
from medevalkit.storage import get_engine, persist_evaluation, persist_question, persist_response, persist_run


def test_compute_aggregates(tmp_path):
    """Test aggregate computation."""
    engine = get_engine(tmp_path / "test.db")
    
    with Session(engine) as session:
        # Create test run
        run = Run(
            id="test-run",
            dataset="test.jsonl",
            n_questions=2,
            responder_model="test-model",
            responder_provider="test",
            judge_model="judge-model",
            judge_provider="test",
            config={},
            started_at=datetime.utcnow(),
        )
        persist_run(session, run)
        
        # Create test questions
        q1 = Question(id="q1", source="test", text="Question 1")
        q2 = Question(id="q2", source="test", text="Question 2")
        persist_question(session, q1)
        persist_question(session, q2)
        
        # Create test responses
        r1 = Response(
            id="r1",
            run_id="test-run",
            question_id="q1",
            answer="Answer 1",
            reasoning="Reasoning 1",
            confidence="high",
            citations=[],
            caveats="None",
            redirect_to_clinician=False,
            raw_completion="{}",
            parse_ok=True,
            latency_ms=100,
            created_at=datetime.utcnow(),
        )
        r2 = Response(
            id="r2",
            run_id="test-run",
            question_id="q2",
            answer="Answer 2",
            reasoning="Reasoning 2",
            confidence="medium",
            citations=[],
            caveats="None",
            redirect_to_clinician=False,
            raw_completion="{}",
            parse_ok=True,
            latency_ms=150,
            created_at=datetime.utcnow(),
        )
        persist_response(session, r1)
        persist_response(session, r2)
        
        # Create test evaluations
        e1 = Evaluation(
            id="e1",
            response_id="r1",
            judge_model="judge-model",
            scores={
                "factual_accuracy": {"score": 5, "rationale": "Good"},
                "safety": {"score": 5, "rationale": "Safe"},
                "hallucination_risk": {"score": 4, "rationale": "Low risk"},
                "calibration": {"score": 5, "rationale": "Well calibrated"},
                "completeness": {"score": 5, "rationale": "Complete"},
            },
            critical_safety_issue=False,
            overall_summary="Good response",
            raw_completion="{}",
            parse_ok=True,
            latency_ms=200,
            created_at=datetime.utcnow(),
        )
        e2 = Evaluation(
            id="e2",
            response_id="r2",
            judge_model="judge-model",
            scores={
                "factual_accuracy": {"score": 3, "rationale": "Okay"},
                "safety": {"score": 2, "rationale": "Some concerns"},
                "hallucination_risk": {"score": 3, "rationale": "Medium risk"},
                "calibration": {"score": 4, "rationale": "Good"},
                "completeness": {"score": 3, "rationale": "Missing details"},
            },
            critical_safety_issue=True,
            critical_safety_rationale="Could harm patient",
            overall_summary="Problematic response",
            raw_completion="{}",
            parse_ok=True,
            latency_ms=250,
            created_at=datetime.utcnow(),
        )
        persist_evaluation(session, e1)
        persist_evaluation(session, e2)
    
    # Compute aggregates
    agg = compute_aggregates("test-run")
    
    # Check results
    assert agg["total_questions"] == 2
    assert agg["successful_responses"] == 2
    assert agg["response_parse_rate"] == 1.0
    assert agg["judge_parse_rate"] == 1.0
    assert agg["critical_safety_issues"] == 1
    
    # Check mean scores
    assert agg["mean_scores"]["factual_accuracy"] == 4.0  # (5+3)/2
    assert agg["mean_scores"]["safety"] == 3.5  # (5+2)/2
    assert agg["mean_scores"]["hallucination_risk"] == 3.5  # (4+3)/2
    assert agg["mean_scores"]["calibration"] == 4.5  # (5+4)/2
    assert agg["mean_scores"]["completeness"] == 4.0  # (5+3)/2
    
    # Check latencies
    assert agg["mean_response_latency_ms"] == 125  # (100+150)/2
    assert agg["mean_judge_latency_ms"] == 225  # (200+250)/2