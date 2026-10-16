"""YOLOv5 model package, extended with project-local AC550 QAT modules."""

from pathlib import Path

_PROJECT_MODELS = Path(__file__).resolve().parents[3] / "models"
if _PROJECT_MODELS.is_dir() and str(_PROJECT_MODELS) not in __path__:
    __path__.append(str(_PROJECT_MODELS))
