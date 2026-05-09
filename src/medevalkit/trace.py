"""Langfuse tracing integration for MedEvalKit."""

import os
from contextlib import contextmanager
from typing import Any, Optional

# Only import Langfuse if available
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False


def get_langfuse_client() -> Optional[Any]:
    """Get Langfuse client if environment variables are set."""
    if not LANGFUSE_AVAILABLE:
        return None
    
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    
    if not (public_key and secret_key):
        return None
    
    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
    )


@contextmanager
def trace_llm_call(model: str, provider: str, messages: list[dict]):
    """
    Context manager for tracing LLM calls with Langfuse.
    
    Yields a dict with trace_id if Langfuse is available, empty dict otherwise.
    """
    client = get_langfuse_client()
    
    if not client:
        yield {}
        return
    
    # Create a new trace
    trace = client.trace(
        name="llm_call",
        metadata={
            "model": model,
            "provider": provider,
        }
    )
    
    # Create generation span
    generation = trace.generation(
        name=f"{provider}/{model}",
        input=messages,
        model=model,
        model_parameters={
            "response_format": {"type": "json_object"},
            "temperature": 0.7,
        }
    )
    
    try:
        yield {"trace_id": trace.id, "generation": generation}
    finally:
        # Flush to ensure events are sent
        client.flush()


def trace_run(run_id: str, dataset: str, config: dict) -> Optional[Any]:
    """Create a trace for an entire evaluation run."""
    client = get_langfuse_client()
    
    if not client:
        return None
    
    return client.trace(
        name="evaluation_run",
        id=run_id,
        metadata={
            "dataset": dataset,
            "config": config,
        }
    )