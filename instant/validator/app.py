"""Local operational HTTP surface for the validator."""

from __future__ import annotations

import asyncio
import contextlib

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .runtime import ValidatorRuntime


def create_app(runtime: ValidatorRuntime, *, run_loop: bool = True) -> FastAPI:
    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        task: asyncio.Task | None = None
        if run_loop:
            await runtime.sync_once()
            task = asyncio.create_task(runtime.run())
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            await runtime.aclose()

    app = FastAPI(
        title="Instant validator",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health() -> dict:
        return runtime.snapshot.to_payload()

    @app.get("/livez")
    async def livez() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        return JSONResponse(
            runtime.snapshot.to_payload(),
            status_code=200 if runtime.snapshot.ready else 503,
        )

    @app.get("/miners")
    async def miners() -> dict:
        return {
            "block": runtime.snapshot.block,
            "netuid": runtime.snapshot.netuid,
            "miners": [miner.to_payload() for miner in runtime.snapshot.miners],
        }

    @app.get("/scores")
    async def scores() -> dict:
        epoch = runtime.state.latest_epoch()
        return {
            "epoch": epoch,
            "scores": [] if epoch is None else runtime.state.scores_for_epoch(epoch),
        }

    return app
