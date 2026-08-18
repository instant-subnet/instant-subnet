# Instant Validator Report v1

This is the frozen Platform-to-Validator contract for Phase 4. The matching JSON fixtures in
`tests/fixtures/` are byte-identical in Platform and Subnet.

## Encoding and authentication

- A report is UTF-8 JSON with exact keys, sorted lexicographically and encoded without optional
  whitespace.
- `digest` is `sha256:` plus the lowercase SHA-256 of the canonical report object after removing
  `digest` and `signature`.
- `signature` is the lowercase 64-byte SR25519 signature of those same canonical payload bytes.
- `signer` is the configured Platform SS58 address. Validators require the configured signer,
  network, and netuid exactly.
- Miner rows are ordered by ascending UID. UIDs and hotkeys are unique. An empty `miners` array is
  valid.

## Finalized epoch

The exact report keys are:

```text
created_at_ms, digest, epoch_end_block, epoch_start_block, finalized_block,
miners, netuid, network, report_id, schema_version, signature, signer, tempo
```

Epoch bounds are inclusive, `epoch_end_block - epoch_start_block + 1 == tempo`, and
`finalized_block >= epoch_end_block`. The Platform publishes one immutable report after the epoch
end is finalized. A terminal attempt appears exactly once, in the first report closed after that
attempt becomes terminal. Pending attempts are not reserved in an earlier report.

Validator configuration defaults to Finney/netuid 46. Phase 4 uses explicit development/local
network flags. The report always binds the selected network and netuid; there is no fallback.

## Miner row

Each row has exactly:

```text
completion_tokens, failed_requests, generation_tps_p50, hotkey,
proof_failed_requests, proof_not_available_requests, proof_timed_out_requests,
proof_verified_requests, routed_requests, successful_requests, uid,
verified_completion_tokens
```

- `successful_requests + failed_requests == routed_requests`.
- All four proof outcome counts sum to `routed_requests`.
- `proof_not_available_requests` covers terminal failures that ended before proof evidence existed.
- `completion_tokens` counts all successfully delivered completion tokens.
- `verified_completion_tokens` and `generation_tps_p50` use only successful, proof-verified work.
- Per-request generation TPS is `floor(completion_tokens * 1000 / (total_ms - ttft_ms))` with a
  positive numerator and denominator. P50 is nearest rank over those integer samples.
- Any nonzero failed or timed-out proof disqualifies the Miner for the entire report epoch. Raw
  metrics and component calculations remain present, but final score and normalized weight are
  zero.

## Deterministic scoring

Relative maxima exclude disqualified Miners. Zero maxima produce zero components.

```text
speed_bps = floor(generation_tps_p50 * 10000 / max_generation_tps_p50)
tokens_bps = floor(verified_completion_tokens * 10000 / max_verified_completion_tokens)
requests_bps = floor(successful_requests * 10000 / max_successful_requests)
success_bps = floor(successful_requests * 10000 / routed_requests)

raw_score_bps = floor(
  (60 * speed_bps + 25 * tokens_bps + 10 * requests_bps + 5 * success_bps) / 100
)

score_bps = 0 if disqualified, otherwise raw_score_bps
normalized_weight = 0 if score_bps == 0, otherwise floor(score_bps * 65535 / max_score_bps)
```

Score output is ordered by ascending UID. Phase 4 logs these results and never submits them.
