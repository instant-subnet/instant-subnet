"""Minimal OpenAI-compatible platform-to-miner gateway."""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from ..protocol.epistula import generate_headers
from ..protocol.keys import Signer
from ..protocol.schemas import ChatCompletionRequest
from ..protocol.ss58 import is_valid

_FORWARDED_RESPONSE_HEADERS = {
    "x-instant-receipt",
    "x-instant-receipt-sig",
    "x-instant-request-id",
    "cache-control",
    "x-accel-buffering",
}


@dataclass(slots=True)
class PlatformContext:
    signer: Signer
    miner_url: str
    miner_ss58: str
    http: httpx.AsyncClient

    def __post_init__(self) -> None:
        self.miner_url = self.miner_url.rstrip("/")
        if not is_valid(self.miner_ss58):
            raise ValueError("miner_ss58 must be a valid Bittensor SS58 address")


def create_app(ctx: PlatformContext) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            await ctx.http.aclose()

    app = FastAPI(
        title="Instant platform",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    async def health_payload() -> dict:
        try:
            response = await ctx.http.get(f"{ctx.miner_url}/health")
            reachable = response.status_code == 200
            payload = response.json() if reachable else None
            miner = payload if isinstance(payload, dict) else None
            reachable = miner is not None
        except (httpx.HTTPError, ValueError):
            miner = None
            reachable = False
        ready = bool(miner and miner.get("ready"))
        return {
            "status": "ok" if reachable and ready else "degraded",
            "ready": ready,
            "miner_url": ctx.miner_url,
            "miner_ss58": ctx.miner_ss58,
            "miner": miner,
        }

    @app.get("/health")
    async def health() -> dict:
        return await health_payload()

    @app.get("/livez")
    async def livez() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        payload = await health_payload()
        return JSONResponse(payload, status_code=200 if payload["ready"] else 503)

    @app.get("/v1/models")
    async def models() -> Response:
        try:
            response = await ctx.http.get(f"{ctx.miner_url}/manifest")
            response.raise_for_status()
            manifest = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            return _error(502, "miner_unavailable", str(exc))
        if not isinstance(manifest, dict):
            return _error(502, "invalid_miner_manifest", "expected a JSON object")
        if manifest.get("hotkey") != ctx.miner_ss58:
            return _error(
                502,
                "miner_identity_mismatch",
                f"expected {ctx.miner_ss58}, got {manifest.get('hotkey')}",
            )
        if not isinstance(manifest.get("model_id"), str):
            return _error(502, "invalid_miner_manifest", "model_id is missing")
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": manifest.get("model_id"),
                        "object": "model",
                        "owned_by": manifest.get("hotkey"),
                    }
                ],
            }
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        raw = await request.body()
        try:
            parsed = ChatCompletionRequest(**json.loads(raw))
        except (ValueError, TypeError) as exc:
            return _error(400, "bad_request", str(exc))

        headers = {
            "Content-Type": "application/json",
            **generate_headers(ctx.signer, raw, signed_for=ctx.miner_ss58),
        }
        url = f"{ctx.miner_url}/v1/chat/completions"
        if not parsed.stream:
            try:
                upstream = await ctx.http.post(url, content=raw, headers=headers)
            except httpx.HTTPError as exc:
                return _error(502, "miner_unavailable", str(exc))
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                headers=_response_headers(upstream),
                media_type=_media_type(upstream, "application/json"),
            )

        try:
            outgoing = ctx.http.build_request("POST", url, content=raw, headers=headers)
            upstream = await ctx.http.send(outgoing, stream=True)
        except httpx.HTTPError as exc:
            return _error(502, "miner_unavailable", str(exc))
        if upstream.status_code >= 400:
            body = await upstream.aread()
            response_headers = _response_headers(upstream)
            media_type = _media_type(upstream, "application/json")
            await upstream.aclose()
            return Response(
                content=body,
                status_code=upstream.status_code,
                headers=response_headers,
                media_type=media_type,
            )

        async def relay() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(
            relay(),
            status_code=upstream.status_code,
            headers=_response_headers(upstream),
            media_type=_media_type(upstream, "text/event-stream"),
        )

    return app


def _response_headers(response: httpx.Response) -> dict[str, str]:
    return {
        name: value
        for name, value in response.headers.items()
        if name.lower() in _FORWARDED_RESPONSE_HEADERS
    }


def _media_type(response: httpx.Response, default: str) -> str:
    return response.headers.get("content-type", default).split(";", 1)[0]


def _error(status: int, error: str, detail: str | None = None) -> JSONResponse:
    return JSONResponse({"error": error, "detail": detail}, status_code=status)
