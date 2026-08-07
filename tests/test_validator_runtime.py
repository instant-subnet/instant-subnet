"""Read-only validator plumbing and operations API."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from instant.validator.app import create_app
from instant.validator.runtime import ValidatorRuntime, serving_miners
from instant.validator.state import open_state


class FakeSubtensor:
    def __init__(self, metagraph, *, block: int = 123):
        self.value = metagraph
        self.block = block
        self.closed = False
        self.calls = []

    def metagraph(self, netuid, lite=True):
        self.calls.append((netuid, lite))
        return self.value

    def get_current_block(self):
        return self.block

    def close(self):
        self.closed = True


def graph(hotkeys, permits, axons):
    return SimpleNamespace(hotkeys=hotkeys, validator_permit=permits, axons=axons)


def axon(ip="0.0.0.0", port=0, serving=False):
    return SimpleNamespace(ip=ip, port=port, is_serving=serving)


def test_serving_miners_drops_unannounced_axons():
    metagraph = graph(
        ["validator", "miner-a", "miner-b"],
        [True, False, False],
        [axon(), axon("10.0.0.8", 8091, True), axon("0.0.0.0", 8091, True)],
    )
    miners = serving_miners(metagraph)
    assert len(miners) == 1
    assert miners[0].uid == 1
    assert miners[0].url == "http://10.0.0.8:8091"


@pytest.mark.asyncio
async def test_sync_reports_registration_permit_miners_and_platform(validator_key):
    platform = FastAPI()

    @platform.get("/health")
    async def health():
        return {"status": "ok"}

    metagraph = graph(
        [validator_key.ss58_address, "miner-a"],
        [True, False],
        [axon(), axon("10.0.0.8", 8091, True)],
    )
    chain = FakeSubtensor(metagraph)
    http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=platform), base_url="http://platform"
    )
    runtime = ValidatorRuntime(
        subtensor=chain,
        wallet=SimpleNamespace(hotkey=validator_key),
        netuid=5,
        state=open_state(":memory:"),
        platform_url="http://platform",
        http=http,
    )
    snapshot = await runtime.sync_once()
    assert snapshot.chain_connected is True
    assert snapshot.block == 123
    assert snapshot.registered is True
    assert snapshot.validator_permit is True
    assert snapshot.ready is True
    assert snapshot.platform_reachable is True
    assert snapshot.miners[0].hotkey == "miner-a"
    assert chain.calls == [(5, True)]
    await runtime.aclose()
    await http.aclose()
    assert chain.closed is True


@pytest.mark.asyncio
async def test_operations_api_separates_liveness_from_readiness(validator_key):
    chain = FakeSubtensor(graph([], [], []))
    http = httpx.AsyncClient()
    runtime = ValidatorRuntime(
        subtensor=chain,
        wallet=SimpleNamespace(hotkey=validator_key),
        netuid=5,
        state=open_state(":memory:"),
        platform_url="",
        http=http,
    )
    transport = httpx.ASGITransport(app=create_app(runtime, run_loop=False))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://validator"
    ) as client:
        assert (await client.get("/livez")).status_code == 200
        assert (await client.get("/readyz")).status_code == 503
        assert (await client.get("/health")).json()["registered"] is False
        assert (await client.get("/miners")).json()["miners"] == []
        assert (await client.get("/scores")).json() == {"epoch": None, "scores": []}
    await runtime.aclose()
    await http.aclose()


@pytest.mark.asyncio
async def test_old_chain_snapshot_is_not_ready(validator_key):
    runtime = ValidatorRuntime(
        subtensor=FakeSubtensor(graph([], [], [])),
        wallet=SimpleNamespace(hotkey=validator_key),
        netuid=5,
        state=open_state(":memory:"),
        platform_url="",
        refresh_s=1,
    )
    runtime.snapshot = replace(
        runtime.snapshot,
        chain_connected=True,
        registered=True,
        validator_permit=True,
        last_sync_ms=1,
    )
    assert runtime.snapshot.ready is False
    assert runtime.snapshot.to_payload()["stale"] is True
    await runtime.aclose()
