"""C. elegans lineage tree parsing and ancestral path traversal."""

import pandas as pd


def parse_lineage_graph(lineage_df: pd.DataFrame) -> dict[str, str]:
    """Parses a lineage DataFrame into a child-to-parent mapping.
    
    Parameters
    ----------
    lineage_df : pd.DataFrame
        DataFrame with columns ['parent', 'daughter1', 'daughter2'].
        
    Returns
    -------
    parent_map : dict[str, str]
        Dictionary mapping child cell name to its parent cell name.
    """
    parent_map: dict[str, str] = {}
    for _, row in lineage_df.iterrows():
        p = str(row["parent"]).strip() if pd.notna(row["parent"]) else None
        d1 = str(row["daughter1"]).strip() if pd.notna(row["daughter1"]) else None
        d2 = str(row["daughter2"]).strip() if pd.notna(row["daughter2"]) else None
        if p:
            if d1:
                parent_map[d1] = p
            if d2:
                parent_map[d2] = p
    return parent_map


def get_ancestral_path_in_interval(
    cell_name: str,
    parent_map: dict[str, str],
    valid_cells: set[str],
) -> tuple[str, list[str]]:
    """Traces ancestry upward to the earliest valid ancestor within a defined interval.
    
    Parameters
    ----------
    cell_name : str
        Target blastomere name.
    parent_map : dict[str, str]
        Child-to-parent mapping.
    valid_cells : set[str]
        Set of valid cells belonging to the current analysis interval or temporal atlas.
        
    Returns
    -------
    local_root : str
        Earliest valid ancestor cell name.
    ancestor_path : list[str]
        Ordered list of intermediate ancestors from root down to (excluding) cell_name.
    """
    curr = cell_name
    path = []
    while curr in parent_map and parent_map[curr] in valid_cells:
        p = parent_map[curr]
        path.append(p)
        curr = p
    local_root = curr
    return local_root, path[::-1]
