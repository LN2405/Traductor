import json
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "app/model/action_lsp.h5"
LABELS_PATH = "app/model/labels.json"

model = load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    label_map = json.load(f)

inv_map = {v: k for k, v in label_map.items()}
actions = [inv_map[i] for i in range(len(inv_map))]

def normalizar_sequence(sequence):
    max_value = np.max(sequence)
    if max_value != 0:
        sequence = sequence / max_value
    return sequence

def predecir_sequence(sequence):
    sequence = np.array(sequence, dtype=np.float32)

    if len(sequence.shape) == 2:
        sequence = np.expand_dims(sequence, axis=0)

    sequence = normalizar_sequence(sequence)
    res = model.predict(sequence, verbose=0)[0]

    idx = np.argmax(res)
    palabra = inv_map[idx]
    confianza = float(res[idx])

    return palabra, confianza, res
