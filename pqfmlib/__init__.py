"""PQFMLib public API."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility.
    tomllib = None

from pqfmlib.maps.xyz import XYZProjectiveQFM
from pqfmlib.maps.cd_ising import CDIsingProjectiveQFM
from pqfmlib.maps.heisenberg import HeisenbergProjectiveQFM


def _package_version() -> str:
    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if tomllib is not None and pyproject_path.is_file():
        try:
            pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
            return pyproject["project"]["version"]
        except (OSError, KeyError, tomllib.TOMLDecodeError):
            pass

    try:
        return version("pqfmlib")
    except PackageNotFoundError:
        return "0.5.0"


__version__ = _package_version()

__all__ = [
    "__version__",
    "XYZProjectiveQFM",
    "CDIsingProjectiveQFM",
    "HeisenbergProjectiveQFM",
]
