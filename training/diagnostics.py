import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model

from training.train import MODEL_PATH, cargar_datos


def diagnosticar_modelo(model, X_train, y_train, label_map):
    print("=" * 50)
    print("DIAGNOSTICO DEL MODELO")
    print("=" * 50)

    actions = list(label_map.keys())

    train_preds = model.predict(X_train, verbose=0)
    train_classes = np.argmax(train_preds, axis=1)
    true_classes = np.argmax(y_train, axis=1)

    cm = confusion_matrix(true_classes, train_classes)

    print("\nMATRIZ DE CONFUSION:")
    for i, action in enumerate(actions):
        confundida = actions[np.argmax(cm[i])]
        marca = "" if cm[i][i] == cm[i].max() else f" confunde con {confundida}"
        print(f"{action:15s}: {cm[i]}{marca}")

    print("\nCONFIANZA POR CLASE:")
    for action in actions:
        idx = label_map[action]
        preds_class = train_preds[true_classes == idx]

        if len(preds_class) == 0:
            continue

        avg_conf = np.mean(preds_class[:, idx])
        estado = "OK" if avg_conf >= 0.80 else "BAJA"
        print(f"{estado} {action:15s}: {avg_conf:.3f}")


def main():
    X_train, _, y_train, _, label_map = cargar_datos()
    model = load_model(MODEL_PATH)
    diagnosticar_modelo(model, X_train, y_train, label_map)


if __name__ == "__main__":
    main()
