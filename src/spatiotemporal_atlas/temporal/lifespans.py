"""Extraction of empirical cell cycle lifespans and midpoints from track coordinates."""

import numpy as np
import pandas as pd


def extract_cell_lifespans(
    pos_df: pd.DataFrame,
    time_col: str = "time",
    embryo_col: str = "series",
    cell_col: str = "cell",
) -> pd.DataFrame:
    """Extracts empirical start, end, duration, and midpoint timestamps per blastomere track.
    
    Parameters
    ----------
    pos_df : pd.DataFrame
        Blastomere positions and timestamps.
    time_col : str, default='time'
        Column with time/frame values.
    embryo_col : str, default='series'
        Column with embryo identifiers.
    cell_col : str, default='cell'
        Column with cell names.
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame containing [embryo_col, cell_col, t_birth, t_divide, duration, t_mid, log_dur].
    """
    cycles = (
        pos_df.groupby([embryo_col, cell_col])[time_col]
        .agg(t_birth="min", t_divide="max")
        .reset_index()
    )
    cycles["duration"] = cycles["t_divide"] - cycles["t_birth"]
    cycles["t_mid"] = 0.5 * (cycles["t_birth"] + cycles["t_divide"])
    df = cycles[cycles["duration"] > 0].copy()
    df["log_dur"] = np.log(df["duration"])
    return df
