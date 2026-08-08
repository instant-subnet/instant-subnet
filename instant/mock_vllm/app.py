"""A deliberately obvious, socket-level mock of the vLLM HTTP surface.

This worker exists for localnet plumbing on CPU hosts. Its model identity is a
fixture identity, never a real Hugging Face model id, so a manifest or receipt
cannot accidentally represent deterministic test output as GPT-OSS inference.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..protocol.schemas import ChatCompletionRequest


def create_app(
    *,
    model_id: str,
    first_token_delay_ms: int = 50,
    token_delay_ms: int = 10,
) -> FastAPI:
    app = FastAPI(
        title="Instant mock vLLM",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.get("/health")
    async def health() -> dict[str, str | bool]:
        return {"status": "ok", "ready": True, "kind": "mock"}

    @app.get("/v1/models")
    async def models() -> dict:
        return {
            "object": "list",
            "data": [{"id": model_id, "object": "model", "owned_by": "instant-mock"}],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            parsed = ChatCompletionRequest(**json.loads(await request.body()))
        except (TypeError, ValueError) as exc:
            return _error(400, "bad_request", str(exc))

        if parsed.model != model_id:
            return _error(
                404,
                "model_not_found",
                f"the mock serves {model_id}, not {parsed.model}",
            )

        chunks = _response_chunks(parsed)
        prompt_tokens = _prompt_tokens(parsed)
        completion_tokens = len(chunks)
        completion_id = f"chatcmpl-mock-{uuid.uuid4().hex}"
        created = int(time.time())
        base = {
            "id": completion_id,
            "created": created,
            "model": model_id,
        }

        if not parsed.stream:
            if first_token_delay_ms:
                await asyncio.sleep(first_token_delay_ms / 1000)
            return JSONResponse(
                {
                    **base,
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "".join(chunks),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": _usage(prompt_tokens, completion_tokens),
                }
            )

        async def stream() -> AsyncIterator[bytes]:
            stream_base = {**base, "object": "chat.completion.chunk"}
            yield _data(
                {
                    **stream_base,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}}],
                }
            )
            if first_token_delay_ms:
                await asyncio.sleep(first_token_delay_ms / 1000)
            for index, chunk in enumerate(chunks):
                if index and token_delay_ms:
                    await asyncio.sleep(token_delay_ms / 1000)
                yield _data(
                    {
                        **stream_base,
                        "choices": [{"index": 0, "delta": {"content": chunk}}],
                    }
                )
            yield _data(
                {
                    **stream_base,
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "stop"}
                    ],
                }
            )
            yield _data(
                {
                    **stream_base,
                    "choices": [],
                    "usage": _usage(prompt_tokens, completion_tokens),
                }
            )
            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def _response_chunks(request: ChatCompletionRequest) -> list[str]:
    last_user = next(
        (
            message.content
            for message in reversed(request.messages)
            if message.role == "user" and isinstance(message.content, str)
        ),
        "",
    )
    excerpt = " ".join(last_user.split())[:120]
    return ["[mock]", " deterministic", " response", f": {excerpt}" if excerpt else "."]


def _prompt_tokens(request: ChatCompletionRequest) -> int:
    return sum(
        max(1, len(message.content.split()))
        for message in request.messages
        if isinstance(message.content, str)
    )


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _data(payload: dict) -> bytes:
    return b"data: " + json.dumps(payload, separators=(",", ":")).encode() + b"\n\n"


def _error(status: int, error: str, detail: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"type": error, "message": detail}}, status_code=status
    )
