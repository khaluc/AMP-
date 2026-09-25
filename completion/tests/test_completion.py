"""Reject corrupt inputs and protect release constraints."""
from pathlib import Path
import math
import pytest
from amp_completion.common import assemble, fasta
from amp_completion.audit import fingerprint_manifest, quantile
from amp_completion.pipeline import main


def test_fasta_rejects_headerless_and_empty_records(tmp_path):
    path = tmp_path / "bad.fasta"
    for text in ["ACDEFGHI\n", ">one\n>two\nACDEFGHI\n", ">\nACDEFGHI\n"]:
        path.write_text(text)
        with pytest.raises(ValueError):
            fasta(path)


def test_assembly_rejects_known_cross_pool_duplicates_and_invalid_sequences():
    result = assemble([["ACDEFGHI", "XXXXXXXX", "AAAAAAAA"],
                       ["AAAAAAAA", "CCCCCCCC"]], {"ACDEFGHI"}, quotas=(1, 1))
    assert result == ["AAAAAAAA", "CCCCCCCC"]
    with pytest.raises(ValueError, match="Insufficient"):
        assemble([["AAAAAAAA"], ["AAAAAAAA"]], set(), quotas=(1, 1))


def test_fingerprint_rejects_drift_and_path_escape(tmp_path):
    path = tmp_path / "data"
    path.write_text("changed")
    with pytest.raises(ValueError, match="Changed"):
        fingerprint_manifest(tmp_path, {"data": "0" * 64})
    with pytest.raises(ValueError, match="escapes"):
        fingerprint_manifest(tmp_path, {"../outside": "0" * 64})


def test_quantile_handles_small_sample_and_rejects_nan():
    assert quantile(range(1, 10)) == 9
    assert math.isinf(quantile([1, 2, 3]))
    with pytest.raises(ValueError):
        quantile([1, math.nan])


def test_entrypoint_refuses_new_seed_without_rescoring(tmp_path):
    with pytest.raises(ValueError, match="frozen seeds"):
        main(["--root", str(tmp_path), "--output", str(tmp_path / "out"), "--seed", "43"])
    assert not (tmp_path / "out").exists()


def test_entrypoint_preserves_existing_results(tmp_path):
    existing = tmp_path / "output"
    existing.mkdir()
    with pytest.raises(ValueError, match="fresh output"):
        main(["--root", str(tmp_path), "--output", str(existing)])


def test_entrypoint_refuses_tampered_verified_output(tmp_path):
    import json
    from amp_completion.common import sha
    output = tmp_path / "generated"
    output.mkdir()
    library = output / "library.fasta"
    top = output / "top.fasta"
    library.write_text(">one\nAAAAAAAA\n")
    top.write_text(">one\nAAAAAAAA\n")
    marker = {"status": "computational_reconstruction_verified", "mode": "cached", "seed": 42,
              "library_sha256": sha(library), "top_sha256": sha(top)}
    (output / "run.json").write_text(json.dumps(marker))
    library.write_text("corrupted")
    with pytest.raises(ValueError, match="Existing output changed"):
        main(["--root", str(tmp_path), "--output", str(output), "--mode", "cached"])
    assert library.read_text() == "corrupted"


def test_entrypoint_preserves_failed_run_for_inspection(tmp_path):
    import json
    output = tmp_path / "generated"
    output.mkdir()
    (output / "run.json").write_text(json.dumps({"status": "failed", "mode": "cached", "seed": 42}))
    with pytest.raises(ValueError, match="incomplete"):
        main(["--root", str(tmp_path), "--output", str(output), "--mode", "cached"])
