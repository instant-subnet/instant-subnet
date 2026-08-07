"""The Instant miner.

A thin, signed, measured proxy in front of vLLM. It does four things:

1. **Authenticate.** Every inference request must carry a valid Epistula v2
   signature from a hotkey on the accept-list — the platform, or a validator
   holding a permit. Nothing else reaches the model.
2. **Serve.** Forward to vLLM, stream the response back token-by-token, and
   get out of the way. The proxy adds one buffer copy and no parsing on the
   hot path.
3. **Attest.** Prove, on demand and against a fresh challenge, that the
   model is running in a TEE on a confidential GPU.
4. **Sign what it served.** Emit a receipt over the request and response
   hashes, so that what the platform reports about this miner can be checked
   against what the miner itself signed.

Deliberately absent: any scoring logic, any opinion about its own
performance, any retry of a failed upstream call. A miner that reasons about
its own score is a miner with an incentive to reason creatively.
"""
