import os
import cv2
import numpy as np

from app.config import DATA_PATH, NO_SEQUENCES, SEQUENCE_LENGTH
from app.services.mediapipe_service import (
    mediapipe_detection,
    draw_styled_landmarks,
    extract_keypoints,
    crear_holistic
)


def obtener_siguiente_sequence(action):
    action_path = os.path.join(DATA_PATH, action)
    os.makedirs(action_path, exist_ok=True)

    existentes = [
        int(folder) for folder in os.listdir(action_path)
        if folder.isdigit()
    ]

    if not existentes:
        return 0

    return max(existentes) + 1


def crear_carpetas(action, inicio):
    for sequence in range(inicio, inicio + NO_SEQUENCES):
        path = os.path.join(DATA_PATH, action, str(sequence))
        os.makedirs(path, exist_ok=True)

    print(f"✅ Carpetas creadas para: {action}")


def grabar_dataset(action, inicio):
    cap = cv2.VideoCapture(0)

    with crear_holistic() as holistic:

        for sequence in range(inicio, inicio + NO_SEQUENCES):
            for frame_num in range(SEQUENCE_LENGTH):

                ret, frame = cap.read()
                if not ret:
                    print("❌ No se pudo leer la cámara")
                    break

                frame = cv2.flip(frame, 1)

                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                if frame_num == 0:
                    cv2.putText(
                        image,
                        f"PREPARATE! Gesto: {action}",
                        (80, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3
                    )
                    cv2.imshow("Grabando", image)
                    cv2.waitKey(2000)

                cv2.putText(
                    image,
                    f"Grabando: {action} - Video {sequence}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    image,
                    f"Frame: {frame_num + 1}/{SEQUENCE_LENGTH}",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                cv2.imshow("Grabando", image)

                keypoints = extract_keypoints(results)

                npy_path = os.path.join(
                    DATA_PATH,
                    action,
                    str(sequence),
                    f"{frame_num}.npy"
                )

                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            print(f"✅ Completado video {sequence} de {action}")

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Grabación terminada")


def verificar_datos(action):
    action_path = os.path.join(DATA_PATH, action)

    sequences = [
        folder for folder in os.listdir(action_path)
        if folder.isdigit()
    ]

    print(f"\n📊 Palabra: {action}")
    print(f"📁 Secuencias guardadas: {len(sequences)}")

    if not sequences:
        print("⚠️ No hay datos todavía")
        return

    sample_path = os.path.join(action_path, sequences[0], "0.npy")
    sample = np.load(sample_path)

    non_zeros = np.count_nonzero(sample)

    print(f"Shape del frame: {sample.shape}")
    print(f"Valores no cero: {non_zeros} de {len(sample)}")


if __name__ == "__main__":
    palabra = input("👉 Ingresa la palabra a grabar: ").strip().lower()

    if not palabra:
        print("❌ Palabra inválida")
    else:
        inicio = obtener_siguiente_sequence(palabra)

        print(f"📌 Palabra: {palabra}")
        print(f"📌 Se grabará desde la secuencia: {inicio}")

        crear_carpetas(palabra, inicio)
        grabar_dataset(palabra, inicio)
        verificar_datos(palabra)