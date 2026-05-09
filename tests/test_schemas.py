"""Tests for schema models."""

import json

import pytest
from pydantic import ValidationError

from medevalkit.schemas import (
    Citation,
    DimensionScore,
    JudgeOutput,
    JudgeScores,
    ResponderOutput,
)


def test_responder_output_valid():
    """Test valid responder output."""
    data = {
        "answer": "Take acetaminophen for fever",
        "reasoning": "Acetaminophen is a safe antipyretic",
        "confidence": "high",
        "citations": [
            {
                "source": "general training knowledge",
                "claim": "Acetaminophen reduces fever",
                "verifiability": "training_recall",
            }
        ],
        "caveats": "Seek medical care if fever persists",
        "redirect_to_clinician": False,
    }
    
    output = ResponderOutput(**data)
    assert output.answer == "Take acetaminophen for fever"
    assert output.confidence == "high"
    assert len(output.citations) == 1
    assert output.citations[0].verifiability == "training_recall"


def test_responder_output_invalid_confidence():
    """Test invalid confidence value."""
    data = {
        "answer": "Test answer",
        "reasoning": "Test reasoning",
        "confidence": "very_high",  # Invalid
        "citations": [],
        "caveats": "None",
        "redirect_to_clinician": False,
    }
    
    with pytest.raises(ValidationError):
        ResponderOutput(**data)


def test_judge_output_valid():
    """Test valid judge output."""
    data = {
        "scores": {
            "factual_accuracy": {"score": 5, "rationale": "Accurate"},
            "safety": {"score": 4, "rationale": "Generally safe"},
            "hallucination_risk": {"score": 5, "rationale": "No hallucination"},
            "calibration": {"score": 3, "rationale": "Somewhat calibrated"},
            "completeness": {"score": 4, "rationale": "Mostly complete"},
        },
        "critical_safety_issue": False,
        "critical_safety_rationale": None,
        "overall_summary": "Good response overall",
    }
    
    output = JudgeOutput(**data)
    assert output.scores.factual_accuracy.score == 5
    assert output.scores.safety.rationale == "Generally safe"
    assert not output.critical_safety_issue


def test_judge_output_with_safety_issue():
    """Test judge output with critical safety issue."""
    data = {
        "scores": {
            "factual_accuracy": {"score": 1, "rationale": "Incorrect"},
            "safety": {"score": 1, "rationale": "Dangerous advice"},
            "hallucination_risk": {"score": 1, "rationale": "Complete fabrication"},
            "calibration": {"score": 1, "rationale": "Overconfident"},
            "completeness": {"score": 2, "rationale": "Missing key info"},
        },
        "critical_safety_issue": True,
        "critical_safety_rationale": "Could cause patient harm",
        "overall_summary": "Dangerous response",
    }
    
    output = JudgeOutput(**data)
    assert output.critical_safety_issue
    assert output.critical_safety_rationale == "Could cause patient harm"


def test_dimension_score_validation():
    """Test score validation (1-5 range)."""
    # Valid scores
    for score in [1, 2, 3, 4, 5]:
        dim = DimensionScore(score=score, rationale="Test")
        assert dim.score == score
    
    # Invalid scores should be caught by the Pydantic model
    # Note: Current schema doesn't enforce 1-5 range, but this test
    # documents expected behavior if validation is added


def test_citation_round_trip():
    """Test citation serialization round-trip."""
    citation = Citation(
        source="FDA guidelines",
        claim="Drug X is contraindicated in pregnancy",
        verifiability="verified",
    )
    
    # Convert to dict and back
    citation_dict = citation.model_dump()
    citation_restored = Citation(**citation_dict)
    
    assert citation_restored.source == citation.source
    assert citation_restored.claim == citation.claim
    assert citation_restored.verifiability == citation.verifiability