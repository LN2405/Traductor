import os
import json

import requests

from app.config import LABELS_PATH


API_VOCABULARY_URL = os.getenv(
    "API_VOCABULARY_URL",
    "https://tec-ii.onrender.com/api/vocabulary"
)


def enviar_vocabulario_a_api(label_map):
    palabras = [
        palabra for palabra in label_map.keys()
        if palabra.strip().lower() != "nada"
    ]

    try:
        response = requests.post(
            API_VOCABULARY_URL,
            json={"palabras": palabras},
            timeout=5
        )
        response.raise_for_status()
        print(f"Vocabulario sincronizado con API: {len(palabras)} palabras")

    except requests.RequestException as exc:
        print(f"No se pudo sincronizar vocabulario con API: {exc}")


def cargar_label_map():
    with open(LABELS_PATH, "r") as f:
        return json.load(f)


def main():
    label_map = cargar_label_map()
    enviar_vocabulario_a_api(label_map)


if __name__ == "__main__":
    main()
