"""The validator: observe, verify, score, set weights.

Four responsibilities, deliberately separated so that only one of them
touches the network in a way that can lie to us:

* ``probe`` — direct probes to the miner and shadow probes through the
  platform. The only honest latency numbers we have.
* ``attest_verify`` — turns an attestation bundle into a yes/no by checking
  the AMD certificate chain and the NVIDIA NRAS token. Produces the
  ``VendorChecks`` that ``instant.protocol.attestation.verify_bundle``
  consumes, so the policy logic stays testable without hardware.
* ``score`` — pure integer arithmetic over observations. No I/O, no clock,
  no floats. Two validators given the same observations must emit the same
  weight vector, byte for byte.
* ``weights`` — the chain write.

``score`` is the consensus-critical one and it is the one with no
dependencies. That is not an accident: everything that decides emissions is
in a module that a test can drive end to end in microseconds.
"""
