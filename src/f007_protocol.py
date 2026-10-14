"""F007 Phase A1 contract checks and fail-closed held-out execution gate.

Structural T006 verification reads manifest metadata, never test images. Model
checkpoint hashes are streamed only when full identity verification is asked
for; no torch model is constructed in this module.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from fractions import Fraction
from pathlib import Path

from src.f002_protocol import verify_t006


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "f007_protocol.json"
PROPOSED = "proposed_pending_team_approval"
ACCEPTED = "accepted_final_protocol"
STATES = ("0", "0.25", "0.50", "0.75", "1.00")
SYSTEMS = ("A", "B", "C", "D", "E", "F")


def _required(record: dict, fields: set[str], context: str) -> None:
    if not isinstance(record, dict) or not fields <= record.keys():
        raise ValueError(f"Incomplete F007 {context}: {sorted(fields - record.keys()) if isinstance(record, dict) else fields}")


def rational(record: dict, context: str) -> Fraction:
    _required(record, {"numerator", "denominator"}, context)
    n, d = record["numerator"], record["denominator"]
    if type(n) is not int or type(d) is not int or d <= 0:
        raise ValueError(f"Invalid exact rational in F007 {context}")
    return Fraction(n, d)


def relative_file(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("F007 path must be a repository-relative POSIX path")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing or unsafe F007 input: {name}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob(root: Path, revision: str, name: str) -> bytes:
    if not re.fullmatch(r"HEAD|[0-9a-f]{40}", revision):
        raise ValueError("Git revision must be HEAD or a full commit SHA")
    try:
        return subprocess.check_output(
            ["git", "show", f"{revision}:{name}"], cwd=root, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"Cannot read canonical Git content: {name}") from exc


def verify_git_lf_file(root: Path, name: str, expected_sha256: str) -> bytes:
    """Hash canonical Git bytes and compare a line-ending-normalized checkout."""
    blob = git_blob(root, "HEAD", name)
    if b"\r" in blob or hashlib.sha256(blob).hexdigest() != expected_sha256:
        raise ValueError(f"Canonical Git LF fingerprint mismatch: {name}")
    local = relative_file(root, name).read_bytes().replace(b"\r\n", b"\n")
    if local != blob:
        raise ValueError(f"Local file differs from canonical Git content: {name}")
    return blob


def validate_structure(protocol: dict) -> None:
    _required(protocol, {"schema_version", "experiment", "status", "accepted_status_value",
                         "authorizes_test_inference", "decision_record", "methodology_details",
                         "frozen_t006", "checkpoints", "quality_requirements", "resource",
                         "resource_states", "systems", "heterogeneous_schedule",
                         "execution_gate", "binary_task"}, "schema")
    if (protocol["schema_version"] != 1 or protocol["experiment"] != "F007" or
            protocol["decision_record"] != "D030" or
            protocol["accepted_status_value"] != ACCEPTED or
            protocol["status"] not in (PROPOSED, ACCEPTED) or
            type(protocol["authorizes_test_inference"]) is not bool or
            protocol["methodology_details"] != "experiments/F007/README.md"):
        raise ValueError("Invalid F007 schema, status or methodology identity")
    frozen = protocol["frozen_t006"]
    _required(frozen, {"canonical_protocol", "canonical_protocol_sha256",
                       "canonical_protocol_sha256_basis", "input_fingerprints_json_pointer",
                       "expected_counts_json_pointer", "test_manifest_sha256_json_pointer",
                       "verification"}, "T006 reference")
    if (frozen["canonical_protocol"] != "configs/f002_protocol.json" or
            frozen["canonical_protocol_sha256_basis"] != "SHA256_of_canonical_Git_LF_blob_bytes" or
            frozen["input_fingerprints_json_pointer"] != "/t006_sha256" or
            frozen["expected_counts_json_pointer"] != "/t006_expected" or
            frozen["test_manifest_sha256_json_pointer"] != "/t006_sha256/test.csv" or
            frozen["verification"] != "src.f002_protocol.verify_t006" or
            not re.fullmatch(r"[0-9a-f]{64}", frozen["canonical_protocol_sha256"])):
        raise ValueError("Invalid canonical T006 reference")
    if tuple(protocol["resource_states"]) != STATES or tuple(protocol["systems"]) != SYSTEMS:
        raise ValueError("F007 six systems or five states changed")
    resource = protocol["resource"]
    _required(resource, {"cmodel_flops", "headroom_ratio", "ccapacity_flops"}, "resource")
    from experiments.F006b.controller import C_MAX
    if (type(resource["cmodel_flops"]) is not int or resource["cmodel_flops"] != C_MAX or
            rational(resource["headroom_ratio"], "headroom") != Fraction(11, 10) or
            rational(resource["ccapacity_flops"], "capacity") != Fraction(11, 10) * C_MAX):
        raise ValueError("F007 exact capacity changed")
    quality = protocol["quality_requirements"]
    _required(quality, {"accuracy_min", "macro_f1_min", "abnormal_recall_min",
                        "validation_profile_path", "validation_profile_sha256"}, "quality")
    if tuple(rational(quality[key], key) for key in
             ("accuracy_min", "macro_f1_min", "abnormal_recall_min")) != (
                Fraction(225, 254), Fraction(44829, 52195), Fraction(57, 76)):
        raise ValueError("F007 quality requirements changed")
    if (quality.get("source") != "F006a_2_frozen_F002b_validation_reference" or
            quality["validation_profile_path"] != "experiments/F006a_2/quality_screen.csv" or
            quality.get("all_three_inclusive") is not True or
            quality.get("clinical_safety_thresholds") is not False or
            not re.fullmatch(r"[0-9a-f]{64}", quality["validation_profile_sha256"])):
        raise ValueError("Invalid F007 validation profile fingerprint")
    checkpoints = protocol["checkpoints"]
    _required(checkpoints, {"f002b_mobilenetv2", "f002c_original_tinyvit",
                            "f005b_nested_tinyvit"}, "checkpoint definitions")
    for name in ("f002b_mobilenetv2", "f002c_original_tinyvit"):
        _required(checkpoints[name], {"path", "sha256", "input_path"}, name)
        if not re.fullmatch(r"[0-9a-f]{64}", checkpoints[name]["sha256"]):
            raise ValueError(f"Invalid F007 {name} checkpoint fingerprint")
    nested = checkpoints["f005b_nested_tinyvit"]
    _required(nested, {"bundle", "checkpoint_sha256", "run_json_sha256",
                       "importance_sha256", "input_path"}, "F005b bundle")
    if any(not re.fullmatch(r"[0-9a-f]{64}", nested[field]) for field in
           ("checkpoint_sha256", "run_json_sha256", "importance_sha256")):
        raise ValueError("Invalid F007 F005b bundle fingerprint")
    if (checkpoints["f002b_mobilenetv2"].get("path") !=
            "checkpoints/f002b_mobilenetv2_seed2026.pt" or
            checkpoints["f002b_mobilenetv2"].get("seed") != 2026 or
            checkpoints["f002b_mobilenetv2"].get("input_path") !=
            "D024_RGB224_bilinear_ImageNet_normalization" or
            checkpoints["f002c_original_tinyvit"].get("path") !=
            "checkpoints/f002c_tinyvit5m_epoch8.pt" or
            checkpoints["f002c_original_tinyvit"].get("selected_epoch") != 8 or
            checkpoints["f002c_original_tinyvit"].get("input_path") !=
            "F003_F004_RGB224_bicubic_ImageNet_normalization" or
            nested.get("bundle") != "checkpoints/f005b_nested_seed2026" or
            nested.get("seed") != 2026 or nested.get("input_path") !=
            "F003_F005_RGB224_bicubic_ImageNet_normalization"):
        raise ValueError("F007 canonical checkpoint path or preprocessing changed")
    definitions = protocol["systems"]
    expected = {
        "A": ("f002b_mobilenetv2", 16, 5107788033),
        "B": ("f002c_original_tinyvit", 16, C_MAX),
        "C": ("f005b_nested_tinyvit", 8, 7266300179),
    }
    for system, (checkpoint, k, flops) in expected.items():
        entry = definitions[system]
        if (entry.get("checkpoint"), entry.get("static_k"), entry.get("flops")) != \
                (checkpoint, k, flops):
            raise ValueError(f"Invalid static F007 system {system}")
    if definitions["C"].get("static_width") != "0.5":
        raise ValueError("F007 C width changed")
    for system, policy in (("D", "minimum_quality"), ("E", "f006a_rule_a"),
                           ("F", "recall_priority")):
        if definitions[system].get("checkpoint") != "f005b_nested_tinyvit" or \
                definitions[system].get("policy") != policy:
            raise ValueError(f"Invalid adaptive F007 system {system}")
    if definitions["D"].get("fixed_point") != "w0.5_k2":
        raise ValueError("F007 minimum-quality point changed")
    schedule = protocol["heterogeneous_schedule"]
    if (schedule.get("unit") != "src.f002_protocol.recording_units_unit_id" or
            schedule.get("salt") != "F007-T013a-v1\n" or
            schedule.get("sort_key") != "SHA256_UTF8_salt_plus_unit_id_then_unit_id" or
            schedule.get("state_by_sorted_rank") != "resource_states[i_mod_5]" or
            schedule.get("same_state_for_all_aliases_of_unit") is not True or
            schedule.get("assignment_uses_labels_or_predictions") is not False):
        raise ValueError("F007 heterogeneous schedule changed")
    binary = protocol["binary_task"]
    if (binary.get("threshold") != "1/2" or binary.get("normal") != 0 or
            binary.get("definite_abnormal") != 1 or
            binary.get("unit_probability") !=
            "arithmetic_mean_of_alias_probabilities_then_threshold"):
        raise ValueError("F007 prediction threshold changed")
    gate = protocol["execution_gate"]
    if (gate.get("requires_authorizes_test_inference_true") is not True or
            gate.get("requires") != ["independent_PR_review",
                "recorded_Charalampos_final_signoff_under_reported_delegation",
                "accepted_committed_F007_protocol", "frozen_hash_validation"]):
        raise ValueError("F007 test-inference gate is incomplete")


def load_protocol(path: Path = PROTOCOL) -> dict:
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_structure(protocol)
    return protocol


def verify_inputs(protocol: dict, *, root: Path = ROOT,
                  check_checkpoints: bool = False) -> dict:
    """Verify tracked metadata first; optionally stream every frozen binary hash."""
    validate_structure(protocol)
    root = Path(root)
    frozen = protocol["frozen_t006"]
    name = frozen["canonical_protocol"]
    blob = verify_git_lf_file(root, name, frozen["canonical_protocol_sha256"])
    f002 = json.loads(blob)
    if not isinstance(f002.get("t006_sha256"), dict) or not f002["t006_sha256"].get("test.csv"):
        raise ValueError("Canonical T006 fingerprint map is incomplete")
    partitions = verify_t006(root / name, root / "data" / "openpocus" / "t006")
    quality = protocol["quality_requirements"]
    verify_git_lf_file(root, quality["validation_profile_path"],
                       quality["validation_profile_sha256"])
    if check_checkpoints:
        checkpoints = protocol["checkpoints"]
        for name in ("f002b_mobilenetv2", "f002c_original_tinyvit"):
            item = checkpoints[name]
            if sha256_file(relative_file(root, item["path"])) != item["sha256"]:
                raise ValueError(f"Frozen {name} checkpoint hash mismatch")
        nested = checkpoints["f005b_nested_tinyvit"]
        for filename, field in (("checkpoint.pt", "checkpoint_sha256"),
                                ("run.json", "run_json_sha256"),
                                ("importance.pt", "importance_sha256")):
            path = relative_file(root, nested["bundle"] + "/" + filename)
            if sha256_file(path) != nested[field]:
                raise ValueError(f"Frozen F005b {filename} hash mismatch")
        run = json.loads(relative_file(root, nested["bundle"] + "/run.json").read_text())
        if (run.get("status"), run.get("mode"), run.get("seed"),
                run.get("checkpoint_sha256")) != (
                "completed", "nested", 2026, nested["checkpoint_sha256"]):
            raise ValueError("Frozen F005b run metadata mismatch")
    return {"protocol_sha256": sha256_file(root / "configs" / "f007_protocol.json"),
            "f002_protocol_git_sha256": frozen["canonical_protocol_sha256"],
            "t006_fingerprints": f002["t006_sha256"],
            "partition_counts": {key: len(value) for key, value in partitions.items()},
            "checkpoint_hashes_checked": check_checkpoints}


def require_test_authorization(protocol: dict, *, root: Path = ROOT,
                               protocol_path: Path | None = None) -> dict:
    """Reject before any held-out image loader can be constructed.

    A future committed approval receipt must point back to a previously
    committed accepted protocol. Keeping the receipt separate avoids the
    impossible requirement that a commit contain its own SHA. CLI flags and
    environment variables are never authorization sources.
    """
    validate_structure(protocol)
    if protocol["status"] != ACCEPTED or protocol["authorizes_test_inference"] is not True:
        raise PermissionError("F007 held-out inference is not authorized")
    root = Path(root)
    protocol_path = Path(protocol_path) if protocol_path else root / "configs/f007_protocol.json"
    receipt_path = root / "configs" / "f007_approval.json"
    try:
        receipt_bytes = receipt_path.read_bytes().replace(b"\r\n", b"\n")
        committed_receipt = git_blob(root, "HEAD", "configs/f007_approval.json")
        if receipt_bytes != committed_receipt or b"\r" in committed_receipt:
            raise ValueError("Approval receipt differs from committed Git content")
        acceptance = json.loads(committed_receipt)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise PermissionError("F007 committed approval receipt is missing or invalid") from exc
    if not isinstance(acceptance, dict) or not all(isinstance(acceptance.get(key), str)
            and acceptance[key] for key in ("protocol_commit", "independent_review_ref",
                                            "charalampos_signoff_ref", "decision_record_ref")):
        raise PermissionError("F007 approval provenance is missing")
    commit = acceptance["protocol_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or \
            acceptance["decision_record_ref"] != "docs/decision_log.md#d030":
        raise PermissionError("F007 accepted commit or D030 reference is invalid")
    try:
        subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                       cwd=root, check=True, capture_output=True)
        committed = git_blob(root, commit, "configs/f007_protocol.json")
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise PermissionError("F007 accepted protocol commit is unverified") from exc
    local = Path(protocol_path).read_bytes().replace(b"\r\n", b"\n")
    if (committed != local or git_blob(root, "HEAD", "configs/f007_protocol.json") != local or
            json.loads(committed) != protocol):
        raise PermissionError("F007 protocol differs from accepted Git commit")
    if not acceptance["independent_review_ref"].startswith(
            "https://github.com/CharalamposGiannakis/adaptive-medical-edge-ai/pull/"):
        raise PermissionError("F007 independent review reference is invalid")
    if not acceptance["charalampos_signoff_ref"].startswith("docs/decision_log.md#d030"):
        raise PermissionError("F007 sign-off reference is invalid")
    decision = git_blob(root, "HEAD", "docs/decision_log.md").decode("utf-8")
    if "### D030" not in decision:
        raise PermissionError("D030 is absent from the committed decision log")
    d030 = decision.split("### D030", 1)[1].split("---", 1)[0]
    if ("**Status:** Accepted" not in d030 or "Charalampos" not in d030 or
            acceptance["independent_review_ref"] not in d030):
        raise PermissionError("D030 accepted sign-off is absent from the committed decision log")
    verification = verify_inputs(protocol, root=root, check_checkpoints=True)
    return {"protocol_commit": commit,
            "accepted_protocol_sha256": hashlib.sha256(committed).hexdigest(),
            "approval_receipt_sha256": hashlib.sha256(committed_receipt).hexdigest(),
            "independent_review_ref": acceptance["independent_review_ref"],
            "charalampos_signoff_ref": acceptance["charalampos_signoff_ref"],
            "decision_record_ref": acceptance["decision_record_ref"],
            "frozen_inputs": verification}
