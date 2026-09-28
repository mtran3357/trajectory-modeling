"""Reference atlas persistence and bundle IO."""

import os
from pathlib import Path
import joblib
from ..types import ReferenceAtlas


def save_atlas(atlas: ReferenceAtlas | dict, filepath: str | Path) -> None:
    """Saves a ReferenceAtlas or atlas dictionary to disk.
    
    Parameters
    ----------
    atlas : ReferenceAtlas or dict
        Atlas bundle.
    filepath : str or Path
        Destination path (.joblib or .pkl).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    payload = atlas.to_dict() if isinstance(atlas, ReferenceAtlas) else atlas
    joblib.dump(payload, filepath, compress=3)


def load_atlas(filepath: str | Path) -> ReferenceAtlas:
    """Loads a ReferenceAtlas bundle from disk.
    
    Parameters
    ----------
    filepath : str or Path
        Path to saved atlas file.
        
    Returns
    -------
    atlas : ReferenceAtlas
        Loaded ReferenceAtlas object.
    """
    raw_dict = joblib.load(filepath)
    return ReferenceAtlas.from_dict(raw_dict)
