import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, default=".data/ReVA_V1")
    parser.add_argument("--dataset-prefix", type=str, default="#dataset")
    parser.add_argument("--answer-format", type=str, choices=["letter", "option_text", "letter_reasoning"], default="letter")
    return parser.parse_args()


def normalize_video_path(raw_path: str, dataset_root: str, dataset_prefix: str) -> str:
    dataset_root_path = Path(dataset_root)
    suffix = raw_path
    if raw_path.startswith(dataset_prefix):
        suffix = raw_path[len(dataset_prefix) :].lstrip("/\\")

    suffix_path = Path(suffix)
    candidates = [suffix_path]
    if len(suffix_path.parts) > 1:
        candidates.append(Path(*suffix_path.parts[1:]))

    for candidate in candidates:
        full_path = dataset_root_path / candidate
        if full_path.exists():
            return candidate.as_posix()

    # Preserve a relative path even if the file is currently unavailable.
    return candidates[-1].as_posix()


def build_prompt(question: str, options: dict[str, str]) -> str:
    labels = sorted(options.keys())
    lines = [question]
    lines.extend(f"{label}. {options[label]}" for label in labels)
    lines.append("Answer with the option's letter from the given choices directly.")
    return "\n".join(lines)


def build_answer(instance: dict, answer_format: str) -> str:
    answer = instance["correct_answer"]
    if answer_format == "letter":
        return answer
    if answer_format == "option_text":
        return instance["options"][answer]
    if answer_format == "letter_reasoning":
        reasoning = instance.get("reasoning", "").strip()
        if reasoning:
            return f"{answer}\n{reasoning}"
        return answer
    raise ValueError(f"Unsupported answer format: {answer_format}")


def convert_question(instance: dict, dataset_root: str, dataset_prefix: str, answer_format: str) -> dict:
    return {
        "id": instance["qa_id"],
        "video": normalize_video_path(instance["file_path"], dataset_root, dataset_prefix),
        "conversations": [
            {"from": "human", "value": build_prompt(instance["question"], instance["options"])},
            {"from": "gpt", "value": build_answer(instance, answer_format)},
        ],
        "metadata": {
            "video_id": instance["video_id"],
            "dataset_name": instance.get("dataset_name"),
            "category": instance.get("category"),
            "subcategory": instance.get("subcategory"),
            "correct_answer": instance.get("correct_answer"),
            "qa_id": instance.get("qa_id"),
        },
    }


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open() as f:
        data = json.load(f)

    questions = data["questions"]
    converted = [
        convert_question(
            instance=instance,
            dataset_root=args.dataset_root,
            dataset_prefix=args.dataset_prefix,
            answer_format=args.answer_format,
        )
        for instance in questions
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(converted)} samples to {output_path}")


if __name__ == "__main__":
    main()
