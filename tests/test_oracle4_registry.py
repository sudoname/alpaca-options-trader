"""Phase 7a tests: champion/challenger registry governance."""
import json
import os

import pytest

from oracle.ml import registry as reg


@pytest.fixture()
def artifact(tmp_path):
    p = tmp_path / "model.bin"
    p.write_bytes(b"model-weights-v1")
    return str(p)


@pytest.fixture()
def rpath(tmp_path):
    return str(tmp_path / "registry.json")


def test_register_defaults_unapproved_challenger(rpath, artifact):
    entry = reg.register_model(rpath, "m1", kind="neural",
                               artifact_path=artifact, notes="unit")
    assert entry["approval_state"] == "unapproved"
    assert entry["role"] == "challenger"
    assert entry["approved_by"] is None
    assert entry["checksums"]["artifact"] == reg.sha256_file(artifact)
    # registry file is valid JSON on disk with history
    on_disk = json.loads(open(rpath, encoding="utf-8").read())
    assert on_disk["models"]["m1"]["model_id"] == "m1"
    assert on_disk["champion_id"] is None
    assert [h["action"] for h in on_disk["history"]] == ["register"]


def test_register_requires_id_kind_and_path(rpath, artifact):
    with pytest.raises(ValueError):
        reg.register_model(rpath, "", kind="neural", artifact_path=artifact)
    with pytest.raises(ValueError):
        reg.register_model(rpath, "m", kind="nonsense", artifact_path=artifact)
    with pytest.raises(ValueError):
        reg.register_model(rpath, "m", kind="neural")  # no paths at all
    with pytest.raises(FileNotFoundError):
        reg.register_model(rpath, "m", kind="neural",
                           artifact_path=str(artifact) + ".missing")


def test_reregister_identical_is_noop_different_rejected(rpath, artifact, tmp_path):
    first = reg.register_model(rpath, "m1", kind="classical",
                               artifact_path=artifact)
    again = reg.register_model(rpath, "m1", kind="classical",
                               artifact_path=artifact)
    assert again["checksums"] == first["checksums"]
    # history should NOT grow on the idempotent no-op
    assert len(reg.load_registry(rpath)["history"]) == 1
    # mutate artifact -> same id must be rejected (immutability)
    with open(artifact, "ab") as f:
        f.write(b"tampered")
    with pytest.raises(ValueError, match="immutable"):
        reg.register_model(rpath, "m1", kind="classical",
                           artifact_path=artifact)


def test_approve_requires_human(rpath, artifact):
    reg.register_model(rpath, "m1", kind="neural", artifact_path=artifact)
    with pytest.raises(ValueError, match="automatic approval"):
        reg.approve_model(rpath, "m1", approved_by="")
    with pytest.raises(ValueError):
        reg.approve_model(rpath, "m1", approved_by="   ")
    with pytest.raises(KeyError):
        reg.approve_model(rpath, "nope", approved_by="alice")
    entry = reg.approve_model(rpath, "m1", approved_by="alice")
    assert entry["approval_state"] == "approved"
    assert entry["approved_by"] == "alice"
    assert entry["approved_utc"]


def test_set_champion_requires_approval_and_demotes_prior(rpath, tmp_path):
    a = tmp_path / "a.bin"; a.write_bytes(b"a")
    b = tmp_path / "b.bin"; b.write_bytes(b"b")
    reg.register_model(rpath, "ma", kind="classical", artifact_path=str(a))
    reg.register_model(rpath, "mb", kind="classical", artifact_path=str(b))

    with pytest.raises(ValueError, match="only approved"):
        reg.set_champion(rpath, "ma", approved_by="alice")
    reg.approve_model(rpath, "ma", approved_by="alice")
    with pytest.raises(ValueError):
        reg.set_champion(rpath, "ma", approved_by="")  # human required

    reg.set_champion(rpath, "ma", approved_by="alice")
    assert reg.get_champion(rpath)["model_id"] == "ma"
    assert reg.get_model(rpath, "ma")["role"] == "champion"

    reg.approve_model(rpath, "mb", approved_by="bob")
    reg.set_champion(rpath, "mb", approved_by="bob")
    assert reg.get_champion(rpath)["model_id"] == "mb"
    assert reg.get_model(rpath, "ma")["role"] == "challenger"  # demoted

    # rollback = set_champion back to prior id (reference change only)
    before = reg.get_model(rpath, "ma")["checksums"]
    reg.set_champion(rpath, "ma", approved_by="alice")
    assert reg.get_champion(rpath)["model_id"] == "ma"
    assert reg.get_model(rpath, "ma")["checksums"] == before  # no file moved

    actions = [h["action"] for h in reg.load_registry(rpath)["history"]]
    assert actions.count("set_champion") == 3
    assert actions.count("demote") == 2


def test_load_missing_is_empty_and_bad_file_rejected(tmp_path):
    empty = reg.load_registry(str(tmp_path / "nope.json"))
    assert empty["models"] == {} and empty["champion_id"] is None
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError):
        reg.load_registry(str(bad))
