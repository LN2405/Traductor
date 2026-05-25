import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from app.config import LOCAL_DATA_PATH, N_FEATURES, SEQUENCE_LENGTH


POSE_SIZE = 33 * 4
HAND_SIZE = 21 * 3
LEFT_HAND_SLICE = slice(POSE_SIZE, POSE_SIZE + HAND_SIZE)
RIGHT_HAND_SLICE = slice(POSE_SIZE + HAND_SIZE, POSE_SIZE + HAND_SIZE * 2)
POSE_SIDE_PAIRS = [
    (1, 4),
    (2, 5),
    (3, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
    (17, 18),
    (19, 20),
    (21, 22),
    (23, 24),
    (25, 26),
    (27, 28),
    (29, 30),
    (31, 32),
]


def parse_actions(value):
    if not value:
        return None
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def ask_value(prompt, default=""):
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return (value or default).strip().strip('"').strip("'")


def ask_yes_no(prompt, default=False):
    default_text = "s" if default else "n"
    value = input(f"{prompt} [s/n, default {default_text}]: ").strip().lower()

    if not value:
        return default

    return value in {"s", "si", "sí", "y", "yes"}


def reduce_features(frame):
    frame = np.ravel(frame)

    if frame.shape[0] == N_FEATURES:
        return frame.astype(np.float32)

    if frame.shape[0] == 1662:
        pose = frame[:POSE_SIZE]
        lh = frame[POSE_SIZE + 1404:POSE_SIZE + 1404 + HAND_SIZE]
        rh = frame[POSE_SIZE + 1404 + HAND_SIZE:POSE_SIZE + 1404 + HAND_SIZE * 2]
        return np.concatenate([pose, lh, rh]).astype(np.float32)

    if frame.shape[0] == 291:
        pose = frame[:POSE_SIZE]
        lh = frame[POSE_SIZE + 33:POSE_SIZE + 33 + HAND_SIZE]
        rh = frame[POSE_SIZE + 33 + HAND_SIZE:POSE_SIZE + 33 + HAND_SIZE * 2]
        return np.concatenate([pose, lh, rh]).astype(np.float32)

    return None


def sequence_dirs(action_path):
    return sorted(
        [path for path in action_path.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )


def load_sequence(sequence_path):
    frames = []

    for frame_num in range(SEQUENCE_LENGTH):
        path = sequence_path / f"{frame_num}.npy"
        if not path.exists():
            return None

        frame = reduce_features(np.load(path))
        if frame is None or frame.shape[0] != N_FEATURES:
            return None

        frames.append(frame)

    return np.array(frames, dtype=np.float32)


def hand_counts(sequence):
    left = 0
    right = 0

    for frame in sequence:
        if np.count_nonzero(frame[LEFT_HAND_SLICE]) > 0:
            left += 1
        if np.count_nonzero(frame[RIGHT_HAND_SLICE]) > 0:
            right += 1

    return left, right


def dominant_hand(sequence):
    left, right = hand_counts(sequence)

    if left == 0 and right == 0:
        return "none"
    if left > 0 and right > 0 and min(left, right) / max(left, right) >= 0.60:
        return "both"
    if left > right:
        return "left"
    if right > left:
        return "right"
    return "both"


def mirror_sequence(sequence):
    mirrored = sequence.copy()

    pose = mirrored[:, :POSE_SIZE].reshape(SEQUENCE_LENGTH, 33, 4)
    pose[:, :, 0] = np.where(pose[:, :, 3] > 0, 1.0 - pose[:, :, 0], pose[:, :, 0])

    for left, right in POSE_SIDE_PAIRS:
        pose[:, [left, right], :] = pose[:, [right, left], :]

    left_hand = mirrored[:, LEFT_HAND_SLICE].copy()
    right_hand = mirrored[:, RIGHT_HAND_SLICE].copy()

    for hand_data in (left_hand, right_hand):
        hand = hand_data.reshape(SEQUENCE_LENGTH, 21, 3)
        hand_mask = np.count_nonzero(hand, axis=2) > 0
        hand[:, :, 0] = np.where(hand_mask, 1.0 - hand[:, :, 0], hand[:, :, 0])

    mirrored[:, LEFT_HAND_SLICE] = right_hand
    mirrored[:, RIGHT_HAND_SLICE] = left_hand

    return mirrored.astype(np.float32)


def load_dataset(root, selected_actions=None):
    root = Path(root)
    data = {}
    skipped = {}

    if not root.exists():
        return data, skipped

    action_paths = [
        path for path in sorted(root.iterdir())
        if path.is_dir() and (selected_actions is None or path.name.lower() in selected_actions)
    ]

    for action_path in action_paths:
        sequences = []
        skipped[action_path.name] = 0

        for sequence_path in sequence_dirs(action_path):
            sequence = load_sequence(sequence_path)
            if sequence is None:
                skipped[action_path.name] += 1
                continue
            sequences.append(sequence)

        if sequences:
            data[action_path.name] = sequences

    return data, skipped


def action_reference(sequences):
    counts = {"left": 0, "right": 0, "both": 0, "none": 0}

    for sequence in sequences:
        counts[dominant_hand(sequence)] += 1

    return max(counts, key=counts.get), counts


def should_mirror(sequence, target_hand):
    current = dominant_hand(sequence)
    return (
        target_hand in {"left", "right"}
        and current in {"left", "right"}
        and current != target_hand
    )


def normalize_sequence(sequence):
    sequence = np.array(sequence, dtype=np.float32)
    max_value = np.max(np.abs(sequence))
    if max_value != 0:
        sequence = sequence / max_value
    return sequence


def sequence_signature(sequence):
    normalized = normalize_sequence(sequence)
    pose = normalized[:, :POSE_SIZE]
    left = normalized[:, LEFT_HAND_SLICE]
    right = normalized[:, RIGHT_HAND_SLICE]

    parts = [
        pose.mean(axis=0),
        pose.std(axis=0),
        left.mean(axis=0),
        left.std(axis=0),
        right.mean(axis=0),
        right.std(axis=0),
    ]

    return np.concatenate(parts).astype(np.float32)


def similarity_profile(reference_sequences, factor):
    if len(reference_sequences) < 2:
        return None

    signatures = np.array([sequence_signature(seq) for seq in reference_sequences])
    center = signatures.mean(axis=0)
    distances = np.linalg.norm(signatures - center, axis=1)
    median = float(np.median(distances))
    percentile = float(np.percentile(distances, 90))
    threshold = max(percentile * factor, median * factor, 0.25)

    return {
        "center": center,
        "threshold": threshold,
        "reference_median_distance": median,
        "reference_p90_distance": percentile,
    }


def sequence_distance(sequence, profile):
    signature = sequence_signature(sequence)
    return float(np.linalg.norm(signature - profile["center"]))


def save_sequence(sequence, output_path):
    output_path.mkdir(parents=True, exist_ok=True)

    for frame_num, frame in enumerate(sequence):
        np.save(output_path / f"{frame_num}.npy", frame)


def export_action(action, sequences, output_root):
    action_output = output_root / action
    if action_output.exists():
        shutil.rmtree(action_output)
    action_output.mkdir(parents=True, exist_ok=True)

    for index, sequence in enumerate(sequences):
        save_sequence(sequence, action_output / str(index))


def next_sequence_index(action_output):
    if not action_output.exists():
        return 0

    existing = [
        int(path.name) for path in action_output.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]

    if not existing:
        return 0

    return max(existing) + 1


def append_action(action, sequences, output_root):
    action_output = output_root / action
    action_output.mkdir(parents=True, exist_ok=True)
    index = next_sequence_index(action_output)

    for sequence in sequences:
        save_sequence(sequence, action_output / str(index))
        index += 1


def main():
    parser = argparse.ArgumentParser(
        description="Compara orientacion de datasets LSP y exporta un dataset corregido."
    )
    parser.add_argument("--reference", default=LOCAL_DATA_PATH, help="Dataset correcto de referencia.")
    parser.add_argument("--incoming", default="", help="Dataset a evaluar/corregir.")
    parser.add_argument(
        "--output",
        default=os.path.join("data", "MP_Data_imported"),
        help="Dataset destino.",
    )
    parser.add_argument("--actions", default="", help="Acciones separadas por coma. Vacio = todas.")
    parser.add_argument(
        "--include-reference",
        dest="include_reference",
        action="store_true",
        default=True,
        help="Incluye tambien las secuencias de referencia en el output.",
    )
    parser.add_argument(
        "--no-include-reference",
        dest="include_reference",
        action="store_false",
        help="Exporta solo el dataset entrante corregido.",
    )
    parser.add_argument("--no-similarity-filter", action="store_true", help="No descarta secuencias alejadas de la referencia.")
    parser.add_argument("--similarity-factor", type=float, default=3.0, help="Que tan permisivo es el filtro de similitud.")
    parser.add_argument("--append", dest="append", action="store_true", default=True, help="Agrega al final sin borrar el destino.")
    parser.add_argument("--rebuild", dest="append", action="store_false", help="Reconstruye el destino desde cero.")
    parser.add_argument("--report-only", action="store_true", help="Solo analiza, no escribe datasets.")
    args = parser.parse_args()

    if len(os.sys.argv) == 1:
        print("\nImportador de datasets LSP")
        print("Deja Enter para usar el valor entre corchetes.\n")
        args.reference = ask_value("Dataset de referencia", args.reference)
        args.incoming = ask_value("Dataset a importar/corregir")
        args.output = ask_value("Dataset destino", args.output)
        args.actions = ask_value("Acciones a procesar separadas por coma; vacio = todas", args.actions)
        args.include_reference = ask_yes_no("Incluir tambien tu dataset de referencia en el destino", True)
        args.append = ask_yes_no("Agregar al destino sin borrar lo existente", True)
        args.report_only = ask_yes_no("Solo generar reporte sin escribir datasets", False)

    args.reference = args.reference.strip().strip('"').strip("'")
    args.incoming = args.incoming.strip().strip('"').strip("'")
    args.output = args.output.strip().strip('"').strip("'")

    selected = parse_actions(args.actions)
    reference_filter = selected if args.append and args.incoming else None
    reference_data, reference_skipped = load_dataset(args.reference, reference_filter)
    incoming_data, incoming_skipped = load_dataset(args.incoming, selected) if args.incoming else ({}, {})

    if args.incoming and not Path(args.incoming).exists():
        raise ValueError(f"El dataset a importar no existe: {args.incoming}")

    if args.incoming and not incoming_data:
        available_actions = [
            path.name for path in sorted(Path(args.incoming).iterdir())
            if path.is_dir()
        ] if Path(args.incoming).exists() else []
        raise ValueError(
            "No se encontraron acciones validas en el dataset externo. "
            f"Acciones disponibles: {available_actions}. "
            "Si querias aceptar el valor por defecto, deja el campo de acciones vacio."
        )

    output_root = Path(args.output)
    export_data = {}
    report = {
        "reference": args.reference,
        "incoming": args.incoming or None,
        "output": args.output,
        "include_reference": args.include_reference,
        "mode": "append" if args.append else "rebuild",
        "augmentation": False,
        "actions": {},
    }

    if args.include_reference and not args.append:
        for action, reference_sequences in reference_data.items():
            export_data[action] = list(reference_sequences)
            report["actions"][action] = {
                "reference_valid": len(reference_sequences),
                "reference_skipped": reference_skipped.get(action, 0),
                "incoming_valid": 0,
                "incoming_skipped": incoming_skipped.get(action, 0),
                "incoming_mirrored": 0,
                "incoming_kept": 0,
                "incoming_rejected_by_similarity": 0,
                "base_exported": len(reference_sequences),
                "exported": len(reference_sequences),
                "local_only": True,
            }

    for action, incoming_sequences in incoming_data.items():
        reference_sequences = reference_data.get(action, [])

        if not reference_sequences:
            report["actions"][action] = {
                "skipped": True,
                "reason": "no_existe_en_referencia_local",
                "incoming_valid": len(incoming_sequences),
                "incoming_skipped": incoming_skipped.get(action, 0),
            }
            print(
                f"{action}: omitida porque no existe en tu dataset local de referencia"
            )
            continue

        target_hand, reference_counts = action_reference(reference_sequences)

        profile = None
        if reference_sequences and not args.no_similarity_filter:
            profile = similarity_profile(reference_sequences, args.similarity_factor)

        fixed_sequences = []
        mirrored = 0
        kept = 0
        rejected_similarity = 0
        accepted_distances = []
        rejected_distances = []
        incoming_counts = {"left": 0, "right": 0, "both": 0, "none": 0}

        for sequence in incoming_sequences:
            current = dominant_hand(sequence)
            incoming_counts[current] += 1

            if should_mirror(sequence, target_hand):
                fixed = mirror_sequence(sequence)
                mirrored += 1
            else:
                fixed = sequence
                kept += 1

            if profile is not None:
                distance = sequence_distance(fixed, profile)
                if distance > profile["threshold"]:
                    rejected_similarity += 1
                    rejected_distances.append(distance)
                    continue
                accepted_distances.append(distance)

            fixed_sequences.append(fixed)

        if args.include_reference and not args.append:
            final_sequences = list(reference_sequences) + fixed_sequences
        else:
            final_sequences = fixed_sequences

        export_data[action] = final_sequences

        fixed_target_counts = {"left": 0, "right": 0, "both": 0, "none": 0}
        for sequence in fixed_sequences:
            fixed_target_counts[dominant_hand(sequence)] += 1

        report["actions"][action] = {
            "reference_valid": len(reference_sequences),
            "reference_skipped": reference_skipped.get(action, 0),
            "reference_dominant": target_hand,
            "reference_counts": reference_counts,
            "incoming_valid": len(incoming_sequences),
            "incoming_skipped": incoming_skipped.get(action, 0),
            "incoming_counts": incoming_counts,
            "incoming_mirrored": mirrored,
            "incoming_kept": kept,
            "incoming_rejected_by_similarity": rejected_similarity,
            "similarity_filter": {
                "enabled": profile is not None,
                "factor": args.similarity_factor,
                "threshold": None if profile is None else profile["threshold"],
                "reference_median_distance": None if profile is None else profile["reference_median_distance"],
                "reference_p90_distance": None if profile is None else profile["reference_p90_distance"],
                "accepted_max_distance": None if not accepted_distances else max(accepted_distances),
                "rejected_min_distance": None if not rejected_distances else min(rejected_distances),
            },
            "fixed_counts": fixed_target_counts,
            "base_exported": len(final_sequences),
            "exported": len(final_sequences),
            "local_only": False,
        }

        print(
            f"{action}: ref={len(reference_sequences)} incoming={len(incoming_sequences)} "
            f"invertidas={mirrored} ya_ok={kept} "
            f"rechazadas={rejected_similarity} exportadas={len(final_sequences)}"
        )

    if not incoming_data:
        print("Sin dataset externo: se exportara solo tu dataset local de referencia.")
        if args.append:
            print("Modo append sin dataset externo: no hay nada nuevo que agregar.")

    if not args.report_only:
        if not export_data:
            print("No hay datos para exportar. No se modifico el destino.")
            return

        output_root.mkdir(parents=True, exist_ok=True)

        for action, sequences in export_data.items():
            if args.append:
                append_action(action, sequences, output_root)
            else:
                export_action(action, sequences, output_root)

        report_path = output_root / "orientation_report.json"
        with open(report_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)

        print(f"\nDataset exportado en: {output_root}")
        print(f"Reporte: {report_path}")
    else:
        print("\nReporte solamente: no se escribieron datasets.")


if __name__ == "__main__":
    main()
