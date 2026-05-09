"""Dataset loading for MedEvalKit."""

import json
from pathlib import Path
from typing import Iterator

from .schemas import Question


def load_jsonl(path: Path) -> Iterator[Question]:
    """
    Load questions from a JSONL file.
    
    Expected format per line:
    {
        "id": "q-123",
        "source": "medqa",
        "category": "cardiology",
        "text": "A 65-year-old patient...",
        "ground_truth": "Answer: B"
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # Validate required fields
                if not data.get("id"):
                    raise ValueError(f"Line {line_num}: Missing 'id' field")
                if not data.get("text"):
                    raise ValueError(f"Line {line_num}: Missing 'text' field")
                
                yield Question(
                    id=data["id"],
                    source=data.get("source", "custom"),
                    category=data.get("category"),
                    text=data["text"],
                    ground_truth=data.get("ground_truth"),
                    meta=data.get("meta", {}),
                )
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {line_num}: Invalid JSON - {e}")
            except Exception as e:
                raise ValueError(f"Line {line_num}: {e}")


def load_dataset(dataset_path: str, limit: int | None = None) -> list[Question]:
    """Load questions from a dataset file, optionally limiting the count."""
    path = Path(dataset_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    questions = []
    for question in load_jsonl(path):
        questions.append(question)
        if limit and len(questions) >= limit:
            break
    
    return questions