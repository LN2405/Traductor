import os
import json
import sys
import builtins
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam

from app.config import SEQUENCE_LENGTH, DATA_PATH, N_FEATURES

MODEL_PATH = "app/model/action_lsp.h5"
LABELS_PATH = "app/model/labels.json"


def print(*args, **kwargs):
    encoding = sys.stdout.encoding or "utf-8"
    safe_args = [
        str(arg).encode(encoding, errors="replace").decode(encoding)
        for arg in args
    ]
    builtins.print(*safe_args, **kwargs)


def reduce_features(res):
    """Convierte keypoints con cara a pose + manos: 258 features."""
    if res.shape[0] == 1662:
        pose = res[:132]
        lh = res[132 + 1404:132 + 1404 + 63]
        rh = res[132 + 1404 + 63:132 + 1404 + 63 + 63]
        return np.concatenate([pose, lh, rh])

    if res.shape[0] == 291:
        pose = res[:132]
        lh = res[132 + 33:132 + 33 + 63]
        rh = res[132 + 33 + 63:132 + 33 + 63 + 63]
        return np.concatenate([pose, lh, rh])

    return res


def obtener_actions():
    return [
        folder for folder in sorted(os.listdir(DATA_PATH))
        if os.path.isdir(os.path.join(DATA_PATH, folder))
    ]

def cargar_datos():
    actions = obtener_actions()

    if len(actions) == 0:
        raise ValueError("No hay palabras grabadas en DATA_PATH")

    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []

    for action in actions:
        action_path = os.path.join(DATA_PATH, action)

        sequences_folders = sorted([
            folder for folder in os.listdir(action_path)
            if folder.isdigit()
        ], key=int)

        for sequence in sequences_folders:
            sequence_path = os.path.join(action_path, sequence)
            window = []

            for frame_num in range(SEQUENCE_LENGTH):
                path = os.path.join(sequence_path, f"{frame_num}.npy")

                try:
                    res = np.load(path)
                    res = reduce_features(res)

                    if res.shape[0] != N_FEATURES:
                        continue

                    if np.count_nonzero(res) < 10:
                        continue

                    window.append(res)

                except Exception:
                    continue

            if len(window) == SEQUENCE_LENGTH:
                sequences.append(window)
                labels.append(label_map[action])

    if len(sequences) == 0:
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
        stratify=labels if len(set(labels)) > 1 else None
    )

    print(f"✅ Palabras detectadas: {actions}")
    print(f"✅ Datos preparados: X={X.shape}, y={y.shape}")
    print(f"   Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test, label_map

def entrenar_modelo(X_train, y_train, X_test, y_test, label_map):
    log_dir = os.path.join("training", "Logs")
    tb_callback = TensorBoard(log_dir=log_dir)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
        verbose=1
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
        metrics=["categorical_accuracy"]
    )

    model.summary()

    model.fit(
        X_train,
        y_train,
        epochs=200,
        validation_split=0.1,
        callbacks=[tb_callback, early_stop]
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"📊 Accuracy en test: {acc*100:.1f}%")
    print(f"📊 Loss en test: {loss:.4f}")

    os.makedirs("app/model", exist_ok=True)

    model.save(MODEL_PATH)
    print("✅ Modelo guardado")

    with open(LABELS_PATH, "w") as f:
        json.dump(label_map, f, indent=4)

    print("✅ Labels guardados")
    
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, label_map = cargar_datos()
    entrenar_modelo(X_train, y_train, X_test, y_test, label_map)
