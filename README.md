# Instant

Instant is a Bittensor subnet focused on fast, reliable, OpenAI-compatible
inference.

The project is under active development and is not yet a production service. This
branch intentionally contains only stable public documentation and repository policy.

## What we are building

- An OpenAI-compatible streaming inference API.
- A miner network optimized for latency, throughput, and reliability.
- Transparent validator measurements and deterministic scoring.
- Secure, operator-friendly software for miners and validators.

## Branches

- `main` is the stable public-facing branch. It contains public documentation and,
  later, reviewed release artifacts.
- `dev` is the active development branch. Code, tests, configuration, and deployment
  work land there first and may change without notice.

Development pull requests should target `dev`. Changes reach `main` only when they are
ready to become part of the stable public surface.

## Documentation

- [Architecture overview](docs/ARCHITECTURE.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)

Public API access and operator onboarding are not open yet. Release and participation
details will be published here when they are ready.
