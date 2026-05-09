"""HTML report generation for MedEvalKit."""

from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from sqlmodel import Session

from .aggregate import compute_aggregates, get_flagged_responses, get_full_results
from .storage import get_engine, get_run


def generate_report(run_id: str) -> Path:
    """Generate HTML report for a run."""
    # Get run metadata
    engine = get_engine()
    with Session(engine) as session:
        run = get_run(session, run_id)
        if not run:
            raise ValueError(f"Run not found: {run_id}")
    
    # Get aggregates and data
    aggregates = compute_aggregates(run_id)
    flagged_responses = get_flagged_responses(run_id)
    full_results = get_full_results(run_id)
    
    # Set up Jinja environment
    template_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("report.html.j2")
    
    # Render template
    html = template.render(
        run=run,
        aggregates=aggregates,
        flagged_responses=flagged_responses,
        full_results=full_results,
        generation_time=datetime.utcnow().isoformat(),
    )
    
    # Create reports directory
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Write report
    report_path = reports_dir / f"{run_id}.html"
    report_path.write_text(html)
    
    return report_path