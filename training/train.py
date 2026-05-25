import builtins
import json
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from app.config import DATA_PATH
from training.data_loader import load_action_sequences, obtener_actions


MODEL_PATH = "app/model/action_lsp.h5"
LABELS_PATH = "app/model/labels.json"


def print(*args, **kwargs):
    encoding = sys.stdout.encoding or "utf-8"
    kwargs.setdefault("flush", True)
    safe_args = [
        str(arg).encode(encoding, errors="replace").decode(encoding)
        for arg in args
    ]
    builtins.print(*safe_args, **kwargs)


def cargar_datos():
    actions = obtener_actions()

    if not actions:
        raise ValueError("No hay palabras grabadas en DATA_PATH")

    print(f"DATA_PATH: {DATA_PATH}")
    print(f"Palabras encontradas: {actions}")

    label_map = {label: index for index, label in enumerate(actions)}
    sequences, labels = [], []

    for action in actions:
        action_sequences = load_action_sequences(action, print_fn=print)
        sequences.extend(action_sequences)
        labels.extend([label_map[action]] * len(action_sequences))

    if not sequences:
        raise ValueError("No se encontraron secuencias completas para entrenar")

    X = np.array(sequences, dtype=np.float32)
    y = to_categorical(labels, num_classes=len(actions)).astype(int)

    max_value = np.max(X)
    if max_value != 0:
        X = X / max_value

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=42,
        stratify=labels if len(set(labels)) > 1 else None,
    )

    print(f"Palabras detectadas: {actions}")
    print(f"Datos preparados: X={X.shape}, y={y.shape}")
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test, label_map


def entrenar_modelo(X_train, y_train, X_test, y_test, label_map):
    log_dir = os.path.join("training", "Logs")
    tb_callback = TensorBoard(log_dir=log_dir)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
        verbose=1,
    )

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(len(label_map), activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )

    model.summary()

    model.fit(
        X_train,
        y_train,
        epochs=200,
        validation_split=0.1,
        callbacks=[tb_callback, early_stop],
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"Accuracy en test: {acc * 100:.1f}%")
    print(f"Loss en test: {loss:.4f}")

    os.makedirs("app/model", exist_ok=True)

    model.save(MODEL_PATH)
    print("Modelo guardado")

    with open(LABELS_PATH, "w", encoding="utf-8") as file:
        json.dump(label_map, file, indent=4)

    print("Labels guardados")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, label_map = cargar_datos()
    entrenar_modelo(X_train, y_train, X_test, y_test, label_map)
