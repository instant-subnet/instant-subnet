"""Config loading, boot-time guards, and small shared utilities.

The guards in :mod:`instant.common.guards` are the reason this package
exists. Every development affordance in Instant — stubbed attestation, GPU
reuse, an unpinned image — is a thing that must never run on mainnet, and
the only reliable way to guarantee that is to make the process refuse to
start rather than to log a warning nobody reads.
"""
