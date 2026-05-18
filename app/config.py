import os

DATA_PATH = os.path.join("data", "MP_Data")

os.makedirs(DATA_PATH, exist_ok=True)

NO_SEQUENCES = 30
SEQUENCE_LENGTH = 30
N_FEATURES = 258

MODEL_PATH = os.path.join("app", "model", "action_lsp.h5")
LABELS_PATH = os.path.join("app", "model", "labels.json")
