"""Color palettes and lineage clade categorization helpers."""

LINEAGE_PALETTE = {
    "AB": "#2980b9",
    "MS": "#27ae60",
    "E": "#d35400",
    "C": "#8e44ad",
    "D": "#16a085",
    "Germline/P": "#c0392b",
    "Other": "#7f8c8d",
}


def get_canonical_clade_color(cell_name: str) -> str:
    """Returns canonical hex color for a given blastomere based on lineage clade."""
    c = str(cell_name).strip()
    if c.startswith("ABa"):
        return "#1f77b4"
    if c.startswith("ABp"):
        return "#2ca02c"
    if c.startswith("AB"):
        return "#17becf"
    if c.startswith("EMS") or c.startswith("MS"):
        return "#ff7f0e"
    if c.startswith("E"):
        return "#2ecc71"
    if c.startswith("C"):
        return "#9467bd"
    if c.startswith("D"):
        return "#8c564b"
    if c.startswith("P") or c.startswith("Z"):
        return "#e377c2"
    return "#7f8c8d"


def assign_lineage(cell: str) -> str:
    """Classifies a cell name into a coarse embryonic lineage clade."""
    c = str(cell).strip()
    if c.startswith("AB"):
        return "AB"
    if c.startswith("MS"):
        return "MS"
    if c.startswith("E"):
        return "E"
    if c.startswith("C"):
        return "C"
    if c.startswith("D"):
        return "D"
    if c.startswith("P") or c.startswith("Z"):
        return "Germline/P"
    return "Other"
