"""Test LLM parse-repair retry path."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from medevalkit.llm import llm_call


class TestOutput(BaseModel):
    """Simple test schema."""
    
    answer: str
    confidence: str


@pytest.mark.asyncio
async def test_parse_success_first_try():
    """Test successful parsing on first attempt."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"answer": "test", "confidence": "high"}'
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    
    with patch("medevalkit.llm.get_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client
        
        parsed, raw, meta = await llm_call(
            system="Test system",
            user="Test user",
            model="test-model",
            provider="ollama",
            schema=TestOutput,
        )
        
        assert parsed is not None
        assert parsed.answer == "test"
        assert parsed.confidence == "high"
        assert meta["parse_ok"] is True
        assert meta["tokens_in"] == 10
        assert meta["tokens_out"] == 20
        
        # Should only call once
        assert mock_client.chat.completions.create.call_count == 1


@pytest.mark.asyncio
async def test_parse_repair_success():
    """Test successful parsing after repair attempt."""
    # First response with invalid JSON
    mock_response1 = MagicMock()
    mock_response1.choices = [MagicMock()]
    mock_response1.choices[0].message.content = "Here is my answer: {not valid json}"
    mock_response1.usage.prompt_tokens = 10
    mock_response1.usage.completion_tokens = 20
    
    # Repair response with valid JSON
    mock_response2 = MagicMock()
    mock_response2.choices = [MagicMock()]
    mock_response2.choices[0].message.content = '{"answer": "repaired", "confidence": "medium"}'
    mock_response2.usage.prompt_tokens = 15
    mock_response2.usage.completion_tokens = 25
    
    with patch("medevalkit.llm.get_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = [mock_response1, mock_response2]
        mock_get_client.return_value = mock_client
        
        parsed, raw, meta = await llm_call(
            system="Test system",
            user="Test user",
            model="test-model",
            provider="ollama",
            schema=TestOutput,
        )
        
        assert parsed is not None
        assert parsed.answer == "repaired"
        assert parsed.confidence == "medium"
        assert meta["parse_ok"] is True
        assert meta["tokens_in"] == 25  # 10 + 15
        assert meta["tokens_out"] == 45  # 20 + 25
        
        # Should call twice (original + repair)
        assert mock_client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_parse_repair_failure():
    """Test when both parse attempts fail."""
    # First response with invalid JSON
    mock_response1 = MagicMock()
    mock_response1.choices = [MagicMock()]
    mock_response1.choices[0].message.content = "Not JSON at all"
    mock_response1.usage.prompt_tokens = 10
    mock_response1.usage.completion_tokens = 20
    
    # Repair response also invalid
    mock_response2 = MagicMock()
    mock_response2.choices = [MagicMock()]
    mock_response2.choices[0].message.content = "Still not JSON"
    mock_response2.usage.prompt_tokens = 15
    mock_response2.usage.completion_tokens = 25
    
    with patch("medevalkit.llm.get_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = [mock_response1, mock_response2]
        mock_get_client.return_value = mock_client
        
        parsed, raw, meta = await llm_call(
            system="Test system",
            user="Test user",
            model="test-model",
            provider="ollama",
            schema=TestOutput,
        )
        
        assert parsed is None
        assert raw == "Still not JSON"  # Returns last attempt
        assert meta["parse_ok"] is False
        assert meta["latency_ms"] > 0
        
        # Should call twice
        assert mock_client.chat.completions.create.call_count == 2