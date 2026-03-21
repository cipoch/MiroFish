#!/usr/bin/env python3
"""
codex-proxy: OpenAI-compatible HTTP proxy for Codex CLI.
Wraps `codex exec --skip-git-repo-check` in a FastAPI server.
"""

import asyncio
import os
import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="codex-proxy")

MAX_WORKERS = int(os.environ.get("CODEX_PROXY_WORKERS", "4"))
CODEX_TIMEOUT = int(os.environ.get("CODEX_PROXY_TIMEOUT", "180"))
semaphore = asyncio.Semaphore(MAX_WORKERS)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = "codex"
    messages: list[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    response_format: Optional[dict] = None


def build_prompt(messages: list[Message], response_format: Optional[dict] = None) -> str:
    parts: list[str] = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"SYSTEM INSTRUCTIONS:\n{msg.content}")
        elif msg.role == "user":
            parts.append(f"USER: {msg.content}")
        elif msg.role == "assistant":
            parts.append(f"ASSISTANT: {msg.content}")

    if response_format and response_format.get("type") == "json_object":
        parts.insert(
            1,
            "IMPORTANT: Respond with valid JSON only. No markdown, no explanation, just pure JSON.",
        )

    return "\n\n".join(parts)


def parse_codex_output(raw: str) -> str:
    """Extract assistant response from codex exec output."""
    parts = raw.split("\ncodex\n")
    if len(parts) > 1:
        content = parts[-1].strip()
        lines = content.split("\n")
        clean = []
        for line in lines:
            if line.strip() == "tokens used":
                break
            clean.append(line)
        return "\n".join(clean).strip()

    return raw.strip()


async def call_codex(prompt: str) -> str:
    """Run codex exec subprocess asynchronously."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "codex",
            "exec",
            "--skip-git-repo-check",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="/tmp",
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=502, detail="Codex CLI error: binary not found") from exc

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=prompt.encode()),
            timeout=CODEX_TIMEOUT,
        )
    except asyncio.TimeoutError as exc:
        proc.kill()
        await proc.wait()
        raise HTTPException(
            status_code=504,
            detail=f"Codex CLI timed out after {CODEX_TIMEOUT}s",
        ) from exc

    if proc.returncode != 0:
        err = stderr.decode()[:300]
        raise HTTPException(status_code=502, detail=f"Codex CLI error: {err}")

    return parse_codex_output(stdout.decode())


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    prompt = build_prompt(req.messages, req.response_format)
    async with semaphore:
        content = await call_codex(prompt)

    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or "codex",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "codex", "object": "model", "created": 0, "owned_by": "openai"}],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "workers": MAX_WORKERS}
