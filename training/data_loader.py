import hashlib
import os

import numpy as np

from app.config import DATA_PATH, N_FEATURES, SEQUENCE_LENGTH


CACHE_ROOT = os.path.join("training", "cache")


def reduce_features(res):
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


def cache_key_for_path(path):
    return hashlib.md5(os.path.abspath(path).encode("utf-8")).hexdigest()[:10]


def action_cache_path(action):
    dataset_key = cache_key_for_path(DATA_PATH)
    return os.path.join(CACHE_ROOT, dataset_key, f"{action}.npz")


def action_signature(action_path):
    npy_count = 0
    latest_mtime = 0.0
    sequence_count = 0

    if not os.path.isdir(action_path):
        return {
            "npy_count": 0,
            "latest_mtime": 0.0,
            "sequence_count": 0,
        }

    for folder in os.listdir(action_path):
        sequence_path = os.path.join(action_path, folder)
        if not folder.isdigit() or not os.path.isdir(sequence_path):
            continue

        sequence_count += 1

        for frame_num in range(SEQUENCE_LENGTH):
            path = os.path.join(sequence_path, f"{frame_num}.npy")
            if not os.path.exists(path):
                continue

            npy_count += 1
            latest_mtime = max(latest_mtime, os.path.getmtime(path))

    return {
        "npy_count": npy_count,
        "latest_mtime": latest_mtime,
        "sequence_count": sequence_count,
    }


def same_signature(left, right):
    return (
        int(left.get("npy_count", -1)) == int(right.get("npy_count", -2))
        and int(left.get("sequence_count", -1)) == int(right.get("sequence_count", -2))
        and abs(float(left.get("latest_mtime", -1)) - float(right.get("latest_mtime", -2))) < 0.0001
    )


def load_action_cache(action, signature):
    cache_path = action_cache_path(action)

    if not os.path.exists(cache_path):
        return None

    try:
        cached = np.load(cache_path, allow_pickle=False)
        cached_signature = {
            "npy_count": int(cached["npy_count"]),
            "latest_mtime": float(cached["latest_mtime"]),
            "sequence_count": int(cached["sequence_count"]),
        }

        if not same_signature(signature, cached_signature):
            return None

        return cached["X"]
    except Exception:
        return None


def save_action_cache(action, signature, action_sequences):
    cache_path = action_cache_path(action)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    np.savez(
        cache_path,
        X=np.array(action_sequences, dtype=np.float32),
        npy_count=int(signature["npy_count"]),
        latest_mtime=float(signature["latest_mtime"]),
        sequence_count=int(signature["sequence_count"]),
    )


def load_action_from_npy(action_path, sequence_folders):
    action_sequences = []

    for sequence in sequence_folders:
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
            action_sequences.append(window)

    return np.array(action_sequences, dtype=np.float32)


def load_action_sequences(action, print_fn=print):
    action_path = os.path.join(DATA_PATH, action)
    sequence_folders = sorted([
        folder for folder in os.listdir(action_path)
        if folder.isdigit()
    ], key=int)

    signature = action_signature(action_path)
    print_fn(f"Cargando {action}: {len(sequence_folders)} secuencias...")

    action_sequences = load_action_cache(action, signature)
    if action_sequences is not None:
        print_fn(f"{action}: cache usado ({len(action_sequences)} secuencias)")
        return action_sequences

    action_sequences = load_action_from_npy(action_path, sequence_folders)
    save_action_cache(action, signature, action_sequences)
    print_fn(f"{action}: cache actualizado ({len(action_sequences)} secuencias)")
    return action_sequences
