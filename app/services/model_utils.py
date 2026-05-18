import json
from app.config import LABELS_PATH

def get_actions():
    with open(LABELS_PATH) as f:
        label_map = json.load(f)

    # invertir mapa
    inv_map = {v: k for k, v in label_map.items()}

    actions = [inv_map[i] for i in range(len(inv_map))]

    return actions