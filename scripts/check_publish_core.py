"""Block dl-azure publishing until its declared core floor is release-safe."""

from importlib import metadata
from pathlib import Path
import tomllib

from packaging.requirements import Requirement
from packaging.version import Version


def main() -> None:
    """Verify release tests use PyPI core and metadata excludes older cores."""
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    lock = tomllib.loads((root / "uv.lock").read_text())
    source = project.get("tool", {}).get("uv", {}).get("sources", {}).get(
        "deep-learning-core", {}
    )
    if "git" in source:
        raise SystemExit(
            "Publishing blocked: remove the dl-core Git source and lock a PyPI release"
        )

    locked_core = next(
        package
        for package in lock["package"]
        if package["name"] == "deep-learning-core"
    )
    registry = locked_core["source"].get("registry", "")
    if not registry.startswith("https://pypi.org/"):
        raise SystemExit("Publishing blocked: the lock must resolve dl-core from PyPI")

    requirement = None
    for dependency in project["project"]["dependencies"]:
        parsed = Requirement(dependency)
        if parsed.name == "deep-learning-core":
            requirement = parsed
            break
    if requirement is None:
        raise SystemExit("Publishing blocked: dl-core dependency is missing")
    floor = max(
        Version(spec.version)
        for spec in requirement.specifier
        if spec.operator in {">=", "=="}
    )
    installed = Version(metadata.version("deep-learning-core"))
    if (
        installed != Version(locked_core["version"])
        or installed not in requirement.specifier
    ):
        raise SystemExit(
            "Publishing blocked: installed dl-core does not match metadata and lock"
        )
    if floor < installed:
        raise SystemExit(
            f"Publishing blocked: metadata permits dl-core {floor}, but tests "
            f"used {installed}; raise the dependency floor or test the floor"
        )

    from dl_core.core import BaseExecutor

    if not all(
        hasattr(BaseExecutor, name)
        for name in ("_classify_run_result", "_after_run_execution")
    ):
        raise SystemExit("Publishing blocked: released dl-core lacks sweep hooks")


if __name__ == "__main__":
    main()
