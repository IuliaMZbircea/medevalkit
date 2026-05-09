"""Aggregate statistics for evaluation runs."""

from collections import defaultdict
from typing import Any

from sqlmodel import Session

from .schemas import Evaluation, Question, Response, Run
from .storage import get_engine, get_evaluations_for_response, get_responses_for_run, get_run


def compute_aggregates(run_id: str) -> dict[str, Any]:
    """Compute aggregate statistics for a run."""
    engine = get_engine()
    
    with Session(engine) as session:
        # Get run
        run = get_run(session, run_id)
        if not run:
            raise ValueError(f"Run not found: {run_id}")
        
        # Get all responses for the run
        responses = get_responses_for_run(session, run_id)
        
        # Initialize aggregates
        total_questions = run.n_questions
        successful_responses = 0
        response_parse_ok = 0
        judge_parse_ok = 0
        critical_safety_issues = 0
        
        # Dimension scores
        dimension_scores = defaultdict(list)
        all_scores = []
        
        # Latency stats
        response_latencies = []
        judge_latencies = []
        
        # Process each response
        for response in responses:
            if response.parse_ok:
                response_parse_ok += 1
            
            response_latencies.append(response.latency_ms)
            
            # Get evaluations for this response
            evaluations = get_evaluations_for_response(session, response.id)
            
            if evaluations:
                successful_responses += 1
                
                for evaluation in evaluations:
                    if evaluation.parse_ok:
                        judge_parse_ok += 1
                    
                    if evaluation.critical_safety_issue:
                        critical_safety_issues += 1
                    
                    judge_latencies.append(evaluation.latency_ms)
                    
                    # Extract dimension scores
                    for dim_name, dim_data in evaluation.scores.items():
                        score = dim_data.get("score", 0)
                        dimension_scores[dim_name].append(score)
                        all_scores.append(score)
        
        # Compute means
        mean_scores = {}
        for dim, scores in dimension_scores.items():
            mean_scores[dim] = sum(scores) / len(scores) if scores else 0.0
        
        # Response and judge parse rates
        response_parse_rate = response_parse_ok / len(responses) if responses else 0.0
        judge_parse_rate = judge_parse_ok / successful_responses if successful_responses > 0 else 0.0
        
        # Latency stats
        mean_response_latency = sum(response_latencies) / len(response_latencies) if response_latencies else 0
        mean_judge_latency = sum(judge_latencies) / len(judge_latencies) if judge_latencies else 0
        
        return {
            "run_id": run_id,
            "total_questions": total_questions,
            "successful_responses": successful_responses,
            "response_parse_rate": response_parse_rate,
            "judge_parse_rate": judge_parse_rate,
            "critical_safety_issues": critical_safety_issues,
            "mean_scores": mean_scores,
            "overall_mean_score": sum(all_scores) / len(all_scores) if all_scores else 0.0,
            "mean_response_latency_ms": mean_response_latency,
            "mean_judge_latency_ms": mean_judge_latency,
        }


def get_flagged_responses(run_id: str) -> list[dict[str, Any]]:
    """Get all responses with critical safety issues."""
    engine = get_engine()
    flagged = []
    
    with Session(engine) as session:
        responses = get_responses_for_run(session, run_id)
        
        for response in responses:
            evaluations = get_evaluations_for_response(session, response.id)
            
            for evaluation in evaluations:
                if evaluation.critical_safety_issue:
                    # Get question
                    question = session.get(Question, response.question_id)
                    
                    flagged.append({
                        "question_id": response.question_id,
                        "question_text": question.text if question else "Unknown",
                        "response_answer": response.answer,
                        "safety_rationale": evaluation.critical_safety_rationale,
                        "overall_summary": evaluation.overall_summary,
                    })
    
    return flagged


def get_full_results(run_id: str) -> list[dict[str, Any]]:
    """Get full results for all questions in a run."""
    engine = get_engine()
    results = []
    
    with Session(engine) as session:
        responses = get_responses_for_run(session, run_id)
        
        for response in responses:
            # Get question
            question = session.get(Question, response.question_id)
            
            # Get evaluations
            evaluations = get_evaluations_for_response(session, response.id)
            
            result = {
                "question_id": response.question_id,
                "question_text": question.text if question else "Unknown",
                "question_category": question.category if question else None,
                "ground_truth": question.ground_truth if question else None,
                "response": {
                    "answer": response.answer,
                    "reasoning": response.reasoning,
                    "confidence": response.confidence,
                    "citations": response.citations,
                    "caveats": response.caveats,
                    "redirect_to_clinician": response.redirect_to_clinician,
                    "parse_ok": response.parse_ok,
                    "latency_ms": response.latency_ms,
                },
                "evaluations": []
            }
            
            for evaluation in evaluations:
                result["evaluations"].append({
                    "scores": evaluation.scores,
                    "critical_safety_issue": evaluation.critical_safety_issue,
                    "critical_safety_rationale": evaluation.critical_safety_rationale,
                    "overall_summary": evaluation.overall_summary,
                    "parse_ok": evaluation.parse_ok,
                    "latency_ms": evaluation.latency_ms,
                })
            
            results.append(result)
    
    return results