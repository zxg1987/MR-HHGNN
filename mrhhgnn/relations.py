from typing import Dict, Tuple

NUM_RELATIONS: int = 12

MODALITIES = ["time", "freq", "img"]

SELF_RELATION_MAP: Dict[str, int] = {"time": 0, "freq": 1, "img": 2}

CROSS_RELATION_MAP: Dict[Tuple[str, str], int] = {
    ("time", "freq"): 6,   ("time", "img"):  7,   ("freq", "img"):  8,
    ("freq", "time"): 9,   ("img",  "time"): 10,  ("img",  "freq"): 11,
}

DEFAULT_RELATION_INDICES: Dict[str, Dict[str, int]] = {
    "spatial":  {"time": 0, "freq": 1, "img": 2},
    "temporal": {"time": 3, "freq": 4, "img": 5},
}
