"""LLM wrapper with parse-repair retry logic."""

import asyncio
import json
import os
import time
from typing import Optional, Type, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from .trace import trace_llm_call

T = TypeVar("T", bound=BaseModel)


def get_client(provider: str) -> AsyncOpenAI:
    """Get OpenAI client configured for the specified provider."""
    if provider == "ollama":
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        return AsyncOpenAI(
            base_url=f"{ollama_host}/v1",
            api_key="ollama",  # Placeholder key for Ollama
        )
    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        return AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
        )
    elif provider == "openai_compat":
        # Generic OpenAI-compatible endpoint
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return AsyncOpenAI(base_url=base_url, api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")


async def llm_call(
    system: str,
    user: str,
    model: str,
    provider: str,
    schema: Type[T],
) -> tuple[Optional[T], str, dict]:
    """
    Call LLM with automatic parse and repair retry.
    
    Returns (parsed_or_none, raw_completion, meta).
    meta = {latency_ms, tokens_in, tokens_out, langfuse_trace_id, parse_ok}
    """
    client = get_client(provider)
    start_time = time.time()
    
    # Initial attempt
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    
    meta = {
        "latency_ms": 0,
        "tokens_in": None,
        "tokens_out": None,
        "langfuse_trace_id": None,
        "parse_ok": False,
    }
    
    # Wrap in Langfuse trace if available
    with trace_llm_call(model, provider, messages) as trace_context:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.7,
            )
            
            raw_completion = response.choices[0].message.content or ""
            
            # Update token counts if available
            if hasattr(response, "usage") and response.usage:
                meta["tokens_in"] = response.usage.prompt_tokens
                meta["tokens_out"] = response.usage.completion_tokens
            
            # Try to parse
            try:
                parsed = schema.model_validate_json(raw_completion)
                meta["parse_ok"] = True
                meta["latency_ms"] = int((time.time() - start_time) * 1000)
                meta["langfuse_trace_id"] = trace_context.get("trace_id")
                return parsed, raw_completion, meta
            except (ValidationError, json.JSONDecodeError) as e:
                # First attempt failed, try repair
                repair_messages = messages + [
                    {
                        "role": "assistant",
                        "content": raw_completion,
                    },
                    {
                        "role": "user",
                        "content": f"Your last output failed JSON validation with: {e}. "
                        f"Return ONLY valid JSON matching the schema. No prose, no markdown fences.",
                    },
                ]
                
                repair_response = await client.chat.completions.create(
                    model=model,
                    messages=repair_messages,
                    response_format={"type": "json_object"},
                    temperature=0.3,  # Lower temp for repair
                )
                
                repair_raw = repair_response.choices[0].message.content or ""
                
                # Update token counts for repair attempt
                if hasattr(repair_response, "usage") and repair_response.usage:
                    meta["tokens_in"] = (meta["tokens_in"] or 0) + repair_response.usage.prompt_tokens
                    meta["tokens_out"] = (meta["tokens_out"] or 0) + repair_response.usage.completion_tokens
                
                try:
                    parsed = schema.model_validate_json(repair_raw)
                    meta["parse_ok"] = True
                    meta["latency_ms"] = int((time.time() - start_time) * 1000)
                    meta["langfuse_trace_id"] = trace_context.get("trace_id")
                    return parsed, repair_raw, meta
                except (ValidationError, json.JSONDecodeError):
                    # Repair failed too
                    meta["latency_ms"] = int((time.time() - start_time) * 1000)
                    meta["langfuse_trace_id"] = trace_context.get("trace_id")
                    return None, repair_raw, meta
                    
        except Exception as e:
            # Network or other errors
            meta["latency_ms"] = int((time.time() - start_time) * 1000)
            meta["langfuse_trace_id"] = trace_context.get("trace_id")
            return None, f"Error calling LLM: {e}", meta