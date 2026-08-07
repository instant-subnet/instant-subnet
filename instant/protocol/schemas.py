"""Request and response models for every Instant endpoint.

These are the shapes in DESIGN.md for the miner, validator, and platform APIs,
expressed once so that the implementations and test suite cannot drift apart.
The platform gateway is Python/FastAPI and will import or generate from these
models rather than maintaining a second hand-written schema definition.

Two conventions run through everything below.

**We are OpenAI-compatible at the edge and strict underneath.** The chat
completion models accept the fields a real OpenAI client sends and ignore
nothing silently: unknown fields are rejected at the miner boundary. Being
permissive here would mean a user's ``temperature`` typo silently producing
different sampling on different miners, which is exactly the sort of
invisible non-determinism that makes a subnet's scores meaningless.

**Nothing that gets signed lives in a pydantic model.** Signed payloads —
receipts, attestation bundles — are plain dataclasses serialised through
:mod:`instant.protocol.canonical`, because pydantic's serialisation is a
moving target across minor versions and a signature that breaks on a
dependency bump is not a signature. Pydantic is for validating untrusted
input at the HTTP boundary. Signing is somewhere else on purpose.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Reject unknown fields everywhere. See the module docstring.
_STRICT = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------
# Inference — the OpenAI-compatible surface
# --------------------------------------------------------------------------


class ChatMessage(BaseModel):
    model_config = _STRICT

    role: Literal["system", "user", "assistant", "tool", "developer"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    """``POST /v1/chat/completions`` on both the miner and the platform.

    The miner receives this from the platform (or from a validator probe) and
    forwards it to vLLM. Keeping the two surfaces identical means a validator
    probe is byte-identical to real user traffic, so a miner cannot serve
    probes better than it serves users — it cannot tell them apart.
    """

    model_config = _STRICT

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    max_tokens: int | None = Field(default=None, ge=1, le=131_072)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=131_072)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    stop: str | list[str] | None = None
    seed: int | None = None
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    n: Literal[1] = 1
    user: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    stream_options: dict[str, Any] | None = None
    # gpt-oss reasoning effort, passed through to the harmony template.
    reasoning_effort: Literal["low", "medium", "high"] | None = None

    @field_validator("stop")
    @classmethod
    def _bounded_stop(cls, v: str | list[str] | None) -> str | list[str] | None:
        if isinstance(v, list) and len(v) > 4:
            raise ValueError("at most 4 stop sequences")
        return v


class Usage(BaseModel):
    model_config = _STRICT

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    model_config = _STRICT

    index: int
    message: ChatMessage
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    model_config = _STRICT

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


# --------------------------------------------------------------------------
# Miner API — DESIGN.md §4
# --------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """``GET /health`` — unauthenticated, cheap, and honest.

    Unauthenticated because a validator needs to distinguish "miner is down"
    from "miner rejected me", and requiring a signature to learn that makes
    the two indistinguishable during exactly the incident where the
    difference matters.

    ``ready`` is false while the model is still loading. A miner that reports
    ready before vLLM has its weights will be probed, fail, and be scored for
    it — so the honest answer is also the profitable one.
    """

    model_config = _STRICT

    status: Literal["ok", "degraded", "loading"]
    ready: bool
    version: str
    model_id: str
    uptime_s: int
    attestation_mode: Literal["hard", "warn", "off"]


class CapacityResponse(BaseModel):
    """``GET /capacity`` — what the miner will accept right now.

    The platform uses this for admission control. A miner that advertises
    capacity it does not have gets routed traffic it then fails to serve,
    which costs it more in score than the traffic was worth. The incentive
    points at accuracy without needing a rule.
    """

    model_config = _STRICT

    max_concurrent: int = Field(ge=0)
    in_flight: int = Field(ge=0)
    queue_depth: int = Field(ge=0)
    max_model_len: int
    accepting: bool
    #: Advertised tokens/sec at current load, integer. Informational only —
    #: never scored, because it is self-reported. The observer measures.
    tokens_per_s_hint: int | None = None


class ManifestResponse(BaseModel):
    """``GET /manifest`` — what this miner is running.

    Cheap to serve, and it is what lets a user or a dashboard answer "which
    model did I actually get" without running a full attestation. The
    ``attestation_id`` links to the last verified bundle, so anyone can go
    from a manifest to the proof behind it.
    """

    model_config = _STRICT

    hotkey: str
    model_id: str
    weights_digest: str
    image_digest: str
    revision: str | None = None
    quantization: str | None = None
    max_model_len: int
    attestation_id: str | None = None
    attested_at_ms: int | None = None


class AttestRequest(BaseModel):
    """``POST /attest`` — a verifier challenges the miner.

    The nonce is chosen by the verifier and must be unpredictable; see
    :func:`instant.protocol.attestation.new_nonce`. The miner cannot answer
    from cache unless it happens to hold a bundle for this exact nonce, which
    is the point.
    """

    model_config = _STRICT

    nonce: str = Field(min_length=64, max_length=66)
    #: Ask the miner to regenerate rather than return a cached bundle. A
    #: fresh nonce already forces this; the flag exists for the case where a
    #: verifier wants to measure how long generation takes.
    force_refresh: bool = False


class AttestResponse(BaseModel):
    """The bundle, as JSON.

    Deliberately typed as a raw dict rather than a pydantic model of the
    bundle: the bytes that get verified are the bytes that arrived, and
    round-tripping them through a model would mean verifying a
    re-serialisation. :meth:`AttestationBundle.from_payload` does the strict
    parse, and it does it on the original payload.
    """

    model_config = _STRICT

    bundle: dict[str, Any]
    generated_ms: int


class ErrorResponse(BaseModel):
    """One error shape everywhere, so clients have one thing to parse."""

    model_config = _STRICT

    error: str
    detail: str | None = None
    request_id: str | None = None


# --------------------------------------------------------------------------
# Platform → validator API — DESIGN.md §5
# --------------------------------------------------------------------------


class MinerStatsWindow(BaseModel):
    """Aggregated observations for one miner over one window.

    Every duration is integer milliseconds and every ratio is integer basis
    points, matching the no-floats rule in
    :mod:`instant.protocol.canonical` — these values are covered by the
    platform's signature over the stats response, and a float would make that
    signature runtime-dependent.
    """

    model_config = _STRICT

    hotkey: str
    uid: int
    requests: int
    successes: int
    failures: int
    ttft_p50_ms: int
    ttft_p95_ms: int
    tokens_per_s_p50: int
    tokens_per_s_p95: int
    success_rate_bps: int = Field(ge=0, le=10_000)
    prompt_tokens: int
    completion_tokens: int
    receipts_seen: int
    receipts_verified: int
    attestation_ok: bool
    attestation_id: str | None = None


class StatsResponse(BaseModel):
    """``GET /validator/v1/stats`` — the platform's report to validators.

    ``receipt_merkle_root`` commits to the receipt set the aggregates were
    computed from, so a validator can sample raw receipts and check that the
    numbers it was handed describe the traffic that actually happened. This
    is what bounds how far the platform can misreport: it can withhold, it
    cannot invent, and any withholding shows up as a merkle mismatch.
    """

    model_config = _STRICT

    window_start_ms: int
    window_end_ms: int
    block_start: int
    block_end: int
    miners: list[MinerStatsWindow]
    receipt_merkle_root: str
    total_requests: int
    generated_at_ms: int


class ReceiptsResponse(BaseModel):
    """``GET /validator/v1/receipts`` — raw receipts for audit sampling."""

    model_config = _STRICT

    window_start_ms: int
    window_end_ms: int
    receipts: list[dict[str, Any]]
    merkle_root: str
    truncated: bool = False


# --------------------------------------------------------------------------
# Miner-facing platform API — DESIGN.md §7.3
# --------------------------------------------------------------------------


class MinerRegisterRequest(BaseModel):
    """``POST /miner/v1/register`` — a miner tells the platform where it is.

    Epistula-signed by the miner hotkey, so the platform does not need a
    separate credential for miners: registration on-chain plus a signature is
    the credential. The platform cross-checks the hotkey against the
    metagraph before it will route anything here.
    """

    model_config = _STRICT

    hotkey: str
    endpoint: str
    port: int = Field(ge=1, le=65_535)
    model_id: str
    max_concurrent: int = Field(ge=1)
    version: str


class MinerRegisterResponse(BaseModel):
    model_config = _STRICT

    accepted: bool
    uid: int | None = None
    reason: str | None = None
    #: How often the platform expects a heartbeat, in seconds.
    heartbeat_interval_s: int = 30
