# Architecture overview

Instant separates customer traffic, inference serving, and network evaluation into
three roles.

```text
Client
  |
  v
Platform gateway ---> Miner ---> Model runtime
                         ^
                         |
                     Validator ---> Bittensor
```

## Platform gateway

The gateway presents an OpenAI-compatible API, selects a miner, and relays streaming
responses. Authentication, quotas, routing, and usage accounting belong at this edge.

## Miner

Miners operate the model-serving hardware. They authenticate machine-to-machine
requests, enforce capacity limits, serve inference, and publish enough information for
validators to discover and evaluate them.

## Validator

Validators observe miner availability and performance, calculate scores
deterministically, and publish weights through Bittensor. Validators are observers of
customer traffic, not an additional request-routing hop.

## Chain

Bittensor coordinates registration, validator authority, weights, and incentives.
Customer prompts and generated responses do not belong on-chain.

The detailed protocol and deployment design remain under active development on the
`dev` branch and will be promoted here as interfaces stabilize.
