# src/labels.py
import numpy as np
from bigearthnet_common.constants import NEW_LABELS, OLD2NEW_LABELS_DICT

# Alphabetical NEW_LABELS is the order the pretrained heads use.
# Confirmed empirically in scripts/sanity_check_labels.py.
CLASSES = list(NEW_LABELS)
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}

_OLD2NEW = {k.strip(): (v.strip() if v else None)
            for k, v in OLD2NEW_LABELS_DICT.items()}

for v in _OLD2NEW.values():
    assert v is None or v in CLASS_TO_IDX, f"mapping target not in NEW_LABELS: {v!r}"


def old_labels_to_multihot(old_labels):
    """43-class label names -> 19-length 0/1 vector. Returns None if nothing survives."""
    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    hit = False
    for name in old_labels:
        key = name.strip()
        if key not in _OLD2NEW:
            raise KeyError(f"unknown BigEarthNet label {key!r}")
        new = _OLD2NEW[key]
        if new is not None:
            vec[CLASS_TO_IDX[new]] = 1.0
            hit = True
    return vec if hit else None