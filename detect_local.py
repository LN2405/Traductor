import cv2
import io
import json
import numpy as np
import pygame
import requests
import threading

from collections import Counter
from gtts import gTTS
from tensorflow.keras.models import load_model

from app.services.mediapipe_service import (
    crear_holistic,
    draw_styled_landmarks,
    extract_keypoints,
    mediapipe_detection,
)


MODEL_PATH = "app/model/action_lsp.h5"
LABELS_PATH = "app/model/labels.json"
API_URL = "https://tec-ii.onrender.com/api/translations"
API_TIMEOUT = 5

SEQUENCE_LENGTH = 30
BUFFER_CONFIRMATION = 10
DEFAULT_THRESHOLD = 0.85
FLIP_CAMERA = True
DRAW_LANDMARKS = True
REQUIRE_HAND = False

thresholds = {
    "nada": 0.90,
}

LETTER_LABELS = set("abcdefghijklmnopqrstuvwxyz")


try:
    pygame.mixer.init()
except Exception:
    pass


try:
    model = load_model(MODEL_PATH)
    print(f"✅ Modelo cargado: {MODEL_PATH}")

    with open(LABELS_PATH, "r") as f:
        label_map = json.load(f)
    print(f"✅ Labels cargados: {LABELS_PATH}")

    inv_map = {v: k for k, v in label_map.items()}
    actions = np.array([inv_map[i] for i in range(len(inv_map))])
    print(f"✅ Clases disponibles: {len(actions)}")
except Exception as e:
    model = None
    actions = np.array([])
    print(f"❌ Error cargando modelo o labels: {e}")


colors = [
    (245, 117, 16),
    (117, 245, 16),
    (16, 117, 245),
    (245, 16, 117),
    (16, 245, 117),
    (200, 80, 200),
    (80, 200, 200),
    (200, 200, 80),
    (255, 128, 0),
    (0, 200, 100),
]


def reproducir_audio(texto):
    try:
        tts = gTTS(text=texto, lang="es", slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)

        pygame.mixer.music.load(fp)
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Error TTS: {e}")


def hablar(texto):
    threading.Thread(target=reproducir_audio, args=(texto,), daemon=True).start()


def enviar_palabra(palabra):
    try:
        requests.post(API_URL, json={"palabra": palabra}, timeout=API_TIMEOUT)
    except Exception as e:
        print("Error enviando a API:", e)


def enviar_palabra_async(palabra):
    threading.Thread(target=enviar_palabra, args=(palabra,), daemon=True).start()


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    row = 0

    for num, prob in enumerate(res):
        if actions[num] == "nada":
            continue

        cv2.rectangle(
            output_frame,
            (0, 60 + row * 35),
            (int(prob * 100), 88 + row * 35),
            colors[num % len(colors)],
            -1,
        )
        cv2.putText(
            output_frame,
            f"{actions[num]}: {prob:.2f}",
            (0, 83 + row * 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        row += 1

    return output_frame


def hay_mano(results):
    return (
        results.left_hand_landmarks is not None
        or results.right_hand_landmarks is not None
    )


def preparar_sequence(sequence):
    input_data = np.array(sequence, dtype=np.float32)
    max_value = np.max(input_data)

    if max_value != 0:
        input_data = input_data / max_value

    return np.expand_dims(input_data, axis=0)


def es_letra(texto):
    return len(texto) == 1 and texto.lower() in LETTER_LABELS


def cerrar_deletreo():
    global letter_buffer, last_spoken

    if not letter_buffer:
        return

    texto = "".join(letter_buffer)
    sentence.append(texto)
    hablar(texto)
    last_spoken = texto
    letter_buffer = []


def procesar_confirmado(texto):
    global last_spoken

    texto = texto.lower()

    if texto == "nada":
        cerrar_deletreo()
        last_spoken = ""
        return

    if texto == last_spoken:
        return

    if es_letra(texto):
        letter_buffer.append(texto)
        enviar_palabra_async(texto)
        last_spoken = texto
        return

    cerrar_deletreo()
    sentence.append(texto)
    enviar_palabra_async(texto)
    hablar(texto)
    last_spoken = texto


sequence = []
sentence = []
buffer = []
letter_buffer = []
last_spoken = ""
res = np.zeros(len(actions))


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

with crear_holistic() as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image, results = mediapipe_detection(frame, holistic)

        if DRAW_LANDMARKS:
            draw_styled_landmarks(image, results)

        if REQUIRE_HAND and not hay_mano(results):
            sequence = []
            buffer = []
            last_spoken = ""
            display_image = cv2.flip(image, 1) if FLIP_CAMERA else image
            cv2.rectangle(display_image, (0, 0), (640, 50), (245, 117, 16), -1)
            cv2.putText(
                display_image,
                " ".join(sentence),
                (3, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("LSP Detector", display_image)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

            continue

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-SEQUENCE_LENGTH:]
        show_probs = False

        if model is not None and len(actions) > 0 and len(sequence) == SEQUENCE_LENGTH:
            expected_features = model.input_shape[-1]
            current_features = len(sequence[-1])

            if current_features == expected_features:
                input_data = preparar_sequence(sequence)
                res = model.predict(input_data, verbose=0)[0]
                pred_idx = np.argmax(res)
                pred_label = actions[pred_idx]
                pred_conf = float(res[pred_idx])
                clase_threshold = thresholds.get(pred_label, DEFAULT_THRESHOLD)

                if pred_conf > clase_threshold:
                    buffer.append(pred_label)

                    if len(buffer) >= BUFFER_CONFIRMATION:
                        palabra = Counter(buffer).most_common(1)[0][0]
                        repeticiones = Counter(buffer)[palabra]

                        if (
                            palabra != last_spoken
                            and repeticiones >= BUFFER_CONFIRMATION - 2
                        ):
                            procesar_confirmado(palabra)

                        buffer = []
                else:
                    buffer = []

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                show_probs = True

        display_image = cv2.flip(image, 1) if FLIP_CAMERA else image

        if show_probs:
            display_image = prob_viz(res, actions, display_image, colors)

        cv2.rectangle(display_image, (0, 0), (640, 50), (245, 117, 16), -1)
        cv2.putText(
            display_image,
            " ".join(sentence + (["".join(letter_buffer)] if letter_buffer else [])),
            (3, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("LSP Detector", display_image)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
print("Deteccion terminada")
