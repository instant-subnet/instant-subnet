"""Canonical JSON must be byte-identical everywhere or signatures are theatre."""

import json

import pytest

from instant.protocol.canonical import (
    NonCanonicalValue,
    bps,
    canonical_json,
    digest,
)


def test_key_order_does_not_affect_output():
    a = {"b": 1, "a": 2, "c": {"z": 1, "y": 2}}
    b = {"c": {"y": 2, "z": 1}, "a": 2, "b": 1}
    assert canonical_json(a) == canonical_json(b)
    assert digest(a) == digest(b)


def test_no_insignificant_whitespace():
    assert canonical_json({"a": 1, "b": [1, 2]}) == b'{"a":1,"b":[1,2]}'


def test_non_ascii_is_not_escaped():
    # JS JSON.stringify does not escape these; ensure_ascii=True would and
    # the two runtimes would disagree on the hash.
    out = canonical_json({"model": "café-Ω"})
    assert out == '{"model":"café-Ω"}'.encode()


def test_floats_are_rejected():
    with pytest.raises(NonCanonicalValue, match="float"):
        canonical_json({"latency": 12.5})
    with pytest.raises(NonCanonicalValue):
        canonical_json({"nested": {"list": [1, 2, 3.0]}})


def test_bool_is_not_treated_as_int():
    assert canonical_json({"ok": True}) == b'{"ok":true}'


def test_unsafe_integers_are_rejected():
    canonical_json({"n": 2**53 - 1})  # fine
    with pytest.raises(NonCanonicalValue, match="safe-integer"):
        canonical_json({"n": 2**53})


def test_non_string_keys_are_rejected():
    with pytest.raises(NonCanonicalValue, match="not a string"):
        canonical_json({1: "a"})


def test_unsupported_types_are_rejected():
    with pytest.raises(NonCanonicalValue, match="unsupported type"):
        canonical_json({"when": object()})


def test_error_message_names_the_path():
    with pytest.raises(NonCanonicalValue, match=r"\$\.a\[1\]\.b"):
        canonical_json({"a": [0, {"b": 1.5}]})


def test_digest_is_prefixed_and_stable():
    d = digest({"a": 1})
    assert d.startswith("sha256:")
    assert len(d) == len("sha256:") + 64
    assert digest({"a": 1}) == d


def test_tuples_serialise_as_arrays():
    assert canonical_json({"a": (1, 2)}) == b'{"a":[1,2]}'


def test_bps_rounds_and_does_not_clamp():
    assert bps(0.9987, 1) == 9987
    assert bps(1, 1) == 10_000
    assert bps(0, 1) == 0
    assert bps(1, 0) == 0
    # A ratio above 1 is a caller bug we want visible, not silently capped.
    assert bps(3, 2) == 15_000


def test_output_is_parseable_json():
    payload = {"a": 1, "b": ["x", "y"], "c": None, "d": True}
    assert json.loads(canonical_json(payload).decode("utf-8")) == payload
