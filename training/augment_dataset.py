import argparse
import json
import os
import random
import re
from pathlib import Path

import numpy as np

from app.config import SEQUENCE_LENGTH
from training.correct_dataset import LEFT_HAND_SLICE, POSE_SIZE, RIGHT_HAND_SLICE, load_dataset, save_sequence


def ask_value(prompt, default=""):
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or default


def ask_yes_no(prompt, default=False):
    default_text = "s" if default else "n"
    value = input(f"{prompt} [s/n, default {default_text}]: ").strip().lower()
    if not value:
        return default
    return value in {"s", "si", "sí", "y", "yes"}


def parse_actions(value):
    if not value:
        return None
    return {
        item.strip().lower()
        for item in re.split(r"[,\s]+", value)
        if item.strip()
    }


def hand_present(frame, hand_slice):
    return np.count_nonzero(frame[hand_slice]) > 0


def valid_source_sequence(action, sequence, min_hand_frames):
    if action == "nada":
        return False

    hand_frames = sum(
        hand_present(frame, LEFT_HAND_SLICE) or hand_present(frame, RIGHT_HAND_SLICE)
        for frame in sequence
    )
    return hand_frames >= min_hand_frames


def add_noise(sequence):
    augmented = sequence.copy().astype(np.float32)

    pose = augmented[:, :POSE_SIZE].reshape(SEQUENCE_LENGTH, 33, 4)
    pose_mask = pose[:, :, 3] > 0
    pose_noise = np.random.normal(0, np.random.uniform(0.002, 0.006), pose[:, :, :3].shape)
    pose[:, :, :3] += pose_noise * pose_mask[:, :, None]

    for hand_slice in (LEFT_HAND_SLICE, RIGHT_HAND_SLICE):
        hand = augmented[:, hand_slice].reshape(SEQUENCE_LENGTH, 21, 3)
        hand_mask = np.count_nonzero(hand, axis=2) > 0
        hand_noise = np.random.normal(0, np.random.uniform(0.002, 0.006), hand.shape)
        hand += hand_noise * hand_mask[:, :, None]

    return augmented


def shift_xy(sequence):
    augmented = sequence.copy().astype(np.float32)
    shift = np.random.uniform(-0.018, 0.018, size=2)

    pose = augmented[:, :POSE_SIZE].reshape(SEQUENCE_LENGTH, 33, 4)
    pose_mask = pose[:, :, 3] > 0
    pose[:, :, :2] += shift * pose_mask[:, :, None]

    for hand_slice in (LEFT_HAND_SLICE, RIGHT_HAND_SLICE):
        hand = augmented[:, hand_slice].reshape(SEQUENCE_LENGTH, 21, 3)
        hand_mask = np.count_nonzero(hand, axis=2) > 0
        hand[:, :, :2] += shift * hand_mask[:, :, None]

    return augmented


def scale_xy(sequence):
    augmented = sequence.copy().astype(np.float32)
    factor = np.random.uniform(0.96, 1.04)
    points = []

    pose = augmented[:, :POSE_SIZE].reshape(SEQUENCE_LENGTH, 33, 4)
    pose_mask = pose[:, :, 3] > 0
    points.append(pose[:, :, :2][pose_mask])

    for hand_slice in (LEFT_HAND_SLICE, RIGHT_HAND_SLICE):
        hand = augmented[:, hand_slice].reshape(SEQUENCE_LENGTH, 21, 3)
        hand_mask = np.count_nonzero(hand, axis=2) > 0
        points.append(hand[:, :, :2][hand_mask])

    points = [item for item in points if len(item)]
    if not points:
        return augmented

    center = np.vstack(points).mean(axis=0)
    pose[:, :, :2] = np.where(
        pose_mask[:, :, None],
        center + (pose[:, :, :2] - center) * factor,
        pose[:, :, :2],
    )

    for hand_slice in (LEFT_HAND_SLICE, RIGHT_HAND_SLICE):
        hand = augmented[:, hand_slice].reshape(SEQUENCE_LENGTH, 21, 3)
        hand_mask = np.count_nonzero(hand, axis=2) > 0
        hand[:, :, :2] = np.where(
            hand_mask[:, :, None],
            center + (hand[:, :, :2] - center) * factor,
            hand[:, :, :2],
        )

    return augmented


def time_warp(sequence):
    factor = np.random.uniform(0.92, 1.08)
    source_indices = np.linspace(0, SEQUENCE_LENGTH - 1, max(2, int(SEQUENCE_LENGTH * factor)))
    warped = []

    for index in source_indices:
        lower = int(np.floor(index))
        upper = min(lower + 1, SEQUENCE_LENGTH - 1)
        alpha = index - lower
        warped.append(sequence[lower] * (1 - alpha) + sequence[upper] * alpha)

    warped = np.array(warped, dtype=np.float32)
    target_indices = np.linspace(0, len(warped) - 1, SEQUENCE_LENGTH)
    resized = []

    for index in target_indices:
        lower = int(np.floor(index))
        upper = min(lower + 1, len(warped) - 1)
        alpha = index - lower
        resized.append(warped[lower] * (1 - alpha) + warped[upper] * alpha)

    return np.array(resized, dtype=np.float32)


def clean_sequence(sequence):
    cleaned = sequence.copy().astype(np.float32)

    pose = cleaned[:, :POSE_SIZE].reshape(SEQUENCE_LENGTH, 33, 4)
    pose[:, :, :2] = np.where(
        pose[:, :, 3:4] > 0,
        np.clip(pose[:, :, :2], 0.0, 1.0),
        pose[:, :, :2],
    )

    for hand_slice in (LEFT_HAND_SLICE, RIGHT_HAND_SLICE):
        hand = cleaned[:, hand_slice].reshape(SEQUENCE_LENGTH, 21, 3)
        hand_mask = np.count_nonzero(hand, axis=2) > 0
        hand[:, :, :2] = np.where(
            hand_mask[:, :, None],
            np.clip(hand[:, :, :2], 0.0, 1.0),
            hand[:, :, :2],
        )

    return cleaned


def augment_sequence(sequence):
    augmented = sequence.copy().astype(np.float32)
    transforms = [add_noise]

    if random.random() < 0.70:
        transforms.append(shift_xy)
    if random.random() < 0.70:
        transforms.append(scale_xy)
    if random.random() < 0.45:
        transforms.append(time_warp)

    random.shuffle(transforms)

    for transform in transforms:
        augmented = transform(augmented)

    return clean_sequence(augmented)


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


def append_generated(action, generated, output_root):
    action_output = output_root / action
    action_output.mkdir(parents=True, exist_ok=True)
    index = next_sequence_index(action_output)

    for sequence in generated:
        save_sequence(sequence, action_output / str(index))
        index += 1


def action_names(root, selected_actions):
    root = Path(root)
    if not root.exists():
        return []

    return [
        path.name for path in sorted(root.iterdir())
        if path.is_dir()
        and path.name != "nada"
        and (selected_actions is None or path.name.lower() in selected_actions)
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Genera variantes de un dataset LSP en una carpeta separada."
    )
    parser.add_argument("--source", default=os.path.join("data", "MP_Data_imported"), help="Dataset final a aumentar.")
    parser.add_argument("--output", default=os.path.join("data", "MP_Data_imported"), help="Dataset final a aumentar.")
    parser.add_argument("--actions", default="", help="Acciones separadas por coma. Vacio = todas.")
    parser.add_argument("--multiplier", type=float, default=2.0, help="Total deseado si incluye originales. 2.0 = doble.")
    parser.add_argument("--min-hand-frames", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    if len(os.sys.argv) == 1:
        print("\nAumentador de datasets LSP")
        print("Deja Enter para usar el valor entre corchetes.\n")
        print(f"Dataset final: {args.source}")
        print(f"Multiplicador total: {args.multiplier}\n")
        args.actions = ask_value("Acciones a procesar separadas por coma; vacio = todas", args.actions)
        args.report_only = ask_yes_no("Solo generar reporte sin escribir datasets", False)

    random.seed(args.seed)
    np.random.seed(args.seed)

    selected_actions = parse_actions(args.actions)
    names = action_names(args.source, selected_actions)
    output_root = Path(args.output)
    report = {
        "source": args.source,
        "output": args.output,
        "multiplier": args.multiplier,
        "mode": "append_generated_only",
        "include_original": False,
        "include_nada": False,
        "actions": {},
    }

    if not names:
        available = action_names(args.source, None)
        raise ValueError(
            "No se encontraron acciones validas en el dataset origen. "
            f"Disponibles: {available}"
        )

    print(f"Cargando {len(names)} acciones...\n")

    for action in names:
        print(f"Procesando {action}...", flush=True)
        data, skipped = load_dataset(args.source, {action})
        sequences = data.get(action, [])
        valid_sources = [
            seq for seq in sequences
            if valid_source_sequence(action, seq, args.min_hand_frames)
        ]

        generated_count = int(round(len(sequences) * (args.multiplier - 1)))

        if args.report_only or not valid_sources:
            generated = []
        else:
            generated = [
                augment_sequence(random.choice(valid_sources))
                for _ in range(generated_count)
            ]

        expected_generated = generated_count if valid_sources else 0
        final_total = len(sequences) + (
            expected_generated if args.report_only else len(generated)
        )

        report["actions"][action] = {
            "source_valid": len(sequences),
            "source_skipped": skipped.get(action, 0),
            "usable_for_augmentation": len(valid_sources),
            "generated": expected_generated if args.report_only else len(generated),
            "final_total": final_total,
        }

        print(
            f"{action}: reales={len(sequences)} usables={len(valid_sources)} "
            f"generadas={report['actions'][action]['generated']} total_final={final_total}"
        )

        if not args.report_only:
            output_root.mkdir(parents=True, exist_ok=True)
            append_generated(action, generated, output_root)

    if not args.report_only:
        report_path = output_root / "augmentation_report.json"
        with open(report_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)
        print(f"\nDataset final aumentado en: {output_root}")
        print(f"Reporte: {report_path}")
    else:
        print("\nReporte solamente: no se escribieron datasets.")


if __name__ == "__main__":
    main()
