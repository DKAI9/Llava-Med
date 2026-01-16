"""Tests for explicit option routing in CLOSED questions."""
from utils.closed_router import build_closed_candidates
from utils.text_norm import normalize_answer


def _route_and_candidates(question: str) -> tuple[str, list[str]]:
    sample = {"question": question, "answer": ""}
    cfg = {"closed_yesno_variants": True, "closed_use_vocab_fallback": False}
    info = build_closed_candidates(sample, [], cfg)
    labels = [normalize_answer(label) for label in info.get("labels", [])]
    return str(info.get("route", "")), labels


def test_route_explicit_or_options() -> None:
    route, labels = _route_and_candidates("Is the modality CT or MRI?")
    assert route == "options"
    assert labels == ["ct", "mri"]


def test_route_slash_options() -> None:
    route, labels = _route_and_candidates("Plane: axial/coronal/sagittal?")
    assert route == "options"
    assert labels == ["axial", "coronal", "sagittal"]


def test_route_versus_options() -> None:
    route, labels = _route_and_candidates("Is it hyperdense vs hypodense?")
    assert route == "options"
    assert labels == ["hyperdense", "hypodense"]


def test_route_yesno_question() -> None:
    route, labels = _route_and_candidates("Is there pleural effusion?")
    assert route == "yesno"
    assert labels == ["yes", "no"]


def test_route_yesno_question_does() -> None:
    route, labels = _route_and_candidates("Does the patient have fracture?")
    assert route == "yesno"
    assert labels == ["yes", "no"]


def test_route_enumeration_options() -> None:
    route, labels = _route_and_candidates("Which is correct, benign, malignant, or normal?")
    assert route == "options"
    assert labels == ["benign", "malignant", "normal"]
