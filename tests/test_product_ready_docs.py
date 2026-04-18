from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_readme_surfaces_checkout_product_ready_boundary() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "CHE-147" in readme
    assert "v0.1 product-ready bar" in readme


def test_product_ready_docs_separate_publication_work() -> None:
    spec = (REPO_ROOT / "SPEC.md").read_text(encoding="utf-8")
    releasing = (REPO_ROOT / "docs" / "releasing.md").read_text(encoding="utf-8")

    assert "checkout-based product-ready" in spec
    assert "outside the v0.1 product-ready" in spec
    assert "CHE-147" in spec
    assert "publication work only" in releasing
    assert "outside that bar" in releasing
    assert "publication ticket" in releasing
