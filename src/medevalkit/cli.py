"""Command-line interface for MedEvalKit."""

import asyncio
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from sqlmodel import Session

from .agents.judge import run_judge
from .agents.responder import run_responder
from .aggregate import compute_aggregates
from .dataset import load_dataset
from .report import generate_report
from .schemas import Evaluation, Question, Response, Run
from .storage import (
    get_engine,
    persist_evaluation,
    persist_question,
    persist_response,
    persist_run,
    get_run,
    get_responses_for_run,
)
from .trace import trace_run

app = typer.Typer(
    name="medeval",
    help="Medical LLM evaluation harness",
    add_completion=False,
)
console = Console()


async def process_question(
    question: Question,
    run: Run,
    session: Session,
    semaphore: asyncio.Semaphore,
) -> tuple[Optional[Response], Optional[Evaluation]]:
    """Process a single question through responder and judge."""
    async with semaphore:
        # Run responder
        responder_output, resp_raw, resp_meta = await run_responder(
            question=question.text,
            model=run.responder_model,
            provider=run.responder_provider,
        )
        
        if responder_output is None:
            console.print(f"[red]✗ Failed to parse response for {question.id}[/red]")
            return None, None
        
        # Create Response record
        response = Response(
            id=str(uuid.uuid4()),
            run_id=run.id,
            question_id=question.id,
            answer=responder_output.answer,
            reasoning=responder_output.reasoning,
            confidence=responder_output.confidence,
            citations=[c.model_dump() for c in responder_output.citations],
            caveats=responder_output.caveats,
            redirect_to_clinician=responder_output.redirect_to_clinician,
            raw_completion=resp_raw,
            parse_ok=resp_meta["parse_ok"],
            latency_ms=resp_meta["latency_ms"],
            tokens_in=resp_meta.get("tokens_in"),
            tokens_out=resp_meta.get("tokens_out"),
            langfuse_trace_id=resp_meta.get("langfuse_trace_id"),
            created_at=datetime.utcnow(),
        )
        persist_response(session, response)
        
        # Run judge
        judge_output, judge_raw, judge_meta = await run_judge(
            question=question.text,
            ground_truth=question.ground_truth,
            response=responder_output,
            model=run.judge_model,
            provider=run.judge_provider,
        )
        
        if judge_output is None:
            console.print(f"[red]✗ Failed to parse evaluation for {question.id}[/red]")
            return response, None
        
        # Convert judge scores to dict format
        scores_dict = {
            "factual_accuracy": {
                "score": judge_output.scores.factual_accuracy.score,
                "rationale": judge_output.scores.factual_accuracy.rationale,
            },
            "safety": {
                "score": judge_output.scores.safety.score,
                "rationale": judge_output.scores.safety.rationale,
            },
            "hallucination_risk": {
                "score": judge_output.scores.hallucination_risk.score,
                "rationale": judge_output.scores.hallucination_risk.rationale,
            },
            "calibration": {
                "score": judge_output.scores.calibration.score,
                "rationale": judge_output.scores.calibration.rationale,
            },
            "completeness": {
                "score": judge_output.scores.completeness.score,
                "rationale": judge_output.scores.completeness.rationale,
            },
        }
        
        # Create Evaluation record
        evaluation = Evaluation(
            id=str(uuid.uuid4()),
            response_id=response.id,
            judge_model=run.judge_model,
            scores=scores_dict,
            critical_safety_issue=judge_output.critical_safety_issue,
            critical_safety_rationale=judge_output.critical_safety_rationale,
            overall_summary=judge_output.overall_summary,
            raw_completion=judge_raw,
            parse_ok=judge_meta["parse_ok"],
            latency_ms=judge_meta["latency_ms"],
            langfuse_trace_id=judge_meta.get("langfuse_trace_id"),
            created_at=datetime.utcnow(),
        )
        persist_evaluation(session, evaluation)
        
        return response, evaluation


@app.command()
def run(
    dataset: str = typer.Option(
        "data/medqa_50.jsonl",
        "--dataset",
        "-d",
        help="Path to dataset JSONL file",
    ),
    n: Optional[int] = typer.Option(
        None,
        "--n",
        "-n",
        help="Number of questions to evaluate (default: all)",
    ),
    responder: str = typer.Option(
        "llama3.1:8b",
        "--responder",
        help="Responder model name",
    ),
    judge: str = typer.Option(
        "qwen2.5:7b-instruct",
        "--judge",
        help="Judge model name",
    ),
    responder_provider: str = typer.Option(
        "ollama",
        "--responder-provider",
        help="Provider for responder model",
    ),
    judge_provider: str = typer.Option(
        "ollama",
        "--judge-provider",
        help="Provider for judge model",
    ),
    concurrency: int = typer.Option(
        2,
        "--concurrency",
        "-c",
        help="Number of concurrent evaluations",
    ),
):
    """Run evaluation on a dataset."""
    # Create run ID
    run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    # Load dataset
    try:
        questions = load_dataset(dataset, limit=n)
        if not questions:
            console.print("[red]No questions found in dataset[/red]")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error loading dataset: {e}[/red]")
        raise typer.Exit(1)
    
    # Create run config
    config = {
        "temperature": 0.7,
        "concurrency": concurrency,
    }
    
    # Print configuration
    console.print("\n[bold]MedEvalKit Run Configuration[/bold]")
    console.print(f"Run ID: {run_id}")
    console.print(f"Dataset: {dataset}")
    console.print(f"Questions: {len(questions)}")
    console.print(f"Responder: {responder} ({responder_provider})")
    console.print(f"Judge: {judge} ({judge_provider})")
    console.print(f"Concurrency: {concurrency}\n")
    
    # Initialize database
    engine = get_engine()
    
    with Session(engine) as session:
        # Create run record
        run_record = Run(
            id=run_id,
            dataset=dataset,
            n_questions=len(questions),
            responder_model=responder,
            responder_provider=responder_provider,
            judge_model=judge,
            judge_provider=judge_provider,
            config=config,
            started_at=datetime.utcnow(),
        )
        persist_run(session, run_record)
        
        # Persist questions
        for q in questions:
            persist_question(session, q)
        
        # Create Langfuse trace if available
        langfuse_trace = trace_run(run_id, dataset, config)
        
        # Process questions with progress bar
        async def process_all():
            semaphore = asyncio.Semaphore(concurrency)
            tasks = []
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Evaluating questions...", total=len(questions))
                
                for question in questions:
                    task_coro = process_question(question, run_record, session, semaphore)
                    tasks.append(task_coro)
                
                results = []
                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    results.append(result)
                    progress.advance(task)
                
                return results
        
        # Run evaluations
        results = asyncio.run(process_all())
        
        # Update run finish time
        run_record.finished_at = datetime.utcnow()
        session.add(run_record)
        session.commit()
    
    # Generate report
    console.print("\n[bold]Generating report...[/bold]")
    report_path = generate_report(run_id)
    
    # Compute and display aggregates
    aggregates = compute_aggregates(run_id)
    
    # Display summary table
    table = Table(title="Evaluation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Questions", str(aggregates["total_questions"]))
    table.add_row("Successful Responses", str(aggregates["successful_responses"]))
    table.add_row("Response Parse Rate", f"{aggregates['response_parse_rate']:.1%}")
    table.add_row("Judge Parse Rate", f"{aggregates['judge_parse_rate']:.1%}")
    table.add_row("Critical Safety Issues", str(aggregates["critical_safety_issues"]))
    
    # Add dimension scores
    for dim, score in aggregates["mean_scores"].items():
        table.add_row(f"Mean {dim.replace('_', ' ').title()}", f"{score:.2f} / 5.0")
    
    console.print("\n")
    console.print(table)
    console.print(f"\n[bold green]✓ Report saved to:[/bold green] {report_path}")


@app.command()
def report(
    run_id: str = typer.Argument(..., help="Run ID to regenerate report for"),
):
    """Regenerate HTML report for an existing run."""
    # Check run exists
    engine = get_engine()
    with Session(engine) as session:
        run = get_run(session, run_id)
        if not run:
            console.print(f"[red]Run not found: {run_id}[/red]")
            raise typer.Exit(1)
    
    # Generate report
    report_path = generate_report(run_id)
    console.print(f"[bold green]✓ Report regenerated:[/bold green] {report_path}")


@app.command()
def compare(
    run_a: str = typer.Argument(..., help="First run ID"),
    run_b: str = typer.Argument(..., help="Second run ID"),
):
    """Compare two evaluation runs."""
    engine = get_engine()
    
    # Get aggregates for both runs
    try:
        agg_a = compute_aggregates(run_a)
        agg_b = compute_aggregates(run_b)
    except Exception as e:
        console.print(f"[red]Error computing aggregates: {e}[/red]")
        raise typer.Exit(1)
    
    # Create comparison table
    table = Table(title=f"Comparison: {run_a} vs {run_b}")
    table.add_column("Metric", style="cyan")
    table.add_column(f"Run A", style="green")
    table.add_column(f"Run B", style="blue")
    table.add_column("Difference", style="yellow")
    
    # Basic metrics
    table.add_row(
        "Total Questions",
        str(agg_a["total_questions"]),
        str(agg_b["total_questions"]),
        "-"
    )
    
    table.add_row(
        "Response Parse Rate",
        f"{agg_a['response_parse_rate']:.1%}",
        f"{agg_b['response_parse_rate']:.1%}",
        f"{(agg_b['response_parse_rate'] - agg_a['response_parse_rate']):.1%}"
    )
    
    table.add_row(
        "Critical Safety Issues",
        str(agg_a["critical_safety_issues"]),
        str(agg_b["critical_safety_issues"]),
        str(agg_b["critical_safety_issues"] - agg_a["critical_safety_issues"])
    )
    
    # Dimension scores
    for dim in agg_a["mean_scores"]:
        score_a = agg_a["mean_scores"][dim]
        score_b = agg_b["mean_scores"][dim]
        diff = score_b - score_a
        
        table.add_row(
            f"{dim.replace('_', ' ').title()}",
            f"{score_a:.2f}",
            f"{score_b:.2f}",
            f"{diff:+.2f}"
        )
    
    console.print("\n")
    console.print(table)


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()