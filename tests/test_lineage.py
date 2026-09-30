"""Unit tests for lineage graph parsing and ancestral interval traversal."""

import pandas as pd
from spatiotemporal_atlas.lineage import (
    parse_lineage_graph,
    get_ancestral_path_in_interval,
)


def test_lineage_graph_traversal():
    """Verify ancestral path and root resolution in a small canonical tree."""
    # ABa -> ABal -> ABala -> ABalaa
    lineage_data = [
        {"parent": "ABa", "daughter1": "ABal", "daughter2": "ABar"},
        {"parent": "ABal", "daughter1": "ABala", "daughter2": "ABalp"},
        {"parent": "ABala", "daughter1": "ABalaa", "daughter2": "ABalap"},
    ]
    df = pd.DataFrame(lineage_data)
    parent_map = parse_lineage_graph(df)

    assert parent_map["ABal"] == "ABa"
    assert parent_map["ABala"] == "ABal"
    assert parent_map["ABalaa"] == "ABala"

    # Interval where ABal is the earliest tracked cell
    valid_cells = {"ABal", "ABala", "ABalaa"}
    root, path = get_ancestral_path_in_interval("ABalaa", parent_map, valid_cells)

    assert root == "ABal"
    assert path == ["ABal", "ABala"]
