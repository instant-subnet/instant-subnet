# Contributing

Instant is still in active development. Please discuss substantial changes with the
maintainers before investing significant implementation time.

## Branch policy

- Target `dev` for application code, tests, configuration, and deployment work.
- Target `main` only for stable public documentation or reviewed release promotion.
- Do not merge development work directly into `main`.

Keep pull requests focused and explain the behavior being changed, how it was tested,
and any operational or security implications.

## Sensitive material

Never commit wallet files, keys, recovery phrases, credentials, private endpoints,
customer data, or populated environment files. Use placeholders in examples and
review the complete diff before pushing.
