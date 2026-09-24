"""Tests for the release-time dl-core compatibility guard."""

from pathlib import Path

import pytest

from scripts import check_publish_core


@pytest.mark.parametrize(
    ("source", "floor", "error"),
    [
        (
            'deep-learning-core = { git = "https://example.com/dl-core.git" }',
            "0.1.8",
            "Git source",
        ),
        ("", "0.1.7", "raise the dependency floor"),
        ("", "0.1.8", None),
    ],
)
def test_publish_guard_requires_released_core_at_declared_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    floor: str,
    error: str | None,
) -> None:
    """The guard rejects development sources and untested lower floors."""
    monkeypatch.setattr(
        check_publish_core, "__file__", str(tmp_path / "scripts" / "check_publish_core.py")
    )
    monkeypatch.setattr(check_publish_core.metadata, "version", lambda name: "0.1.8")
    (tmp_path / "pyproject.toml").write_text(
        "[project]\n"
        f'dependencies = ["deep-learning-core>={floor},<0.2"]\n'
        "[tool.uv.sources]\n"
        f"{source}\n",
        encoding="utf-8",
    )
    (tmp_path / "uv.lock").write_text(
        '[[package]]\nname = "deep-learning-core"\nversion = "0.1.8"\n'
        'source = { registry = "https://pypi.org/simple" }\n',
        encoding="utf-8",
    )

    if error:
        with pytest.raises(SystemExit, match=error):
            check_publish_core.main()
    else:
        check_publish_core.main()
