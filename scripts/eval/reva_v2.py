import argparse
import copy
import json
import os
import re
from pathlib import Path
from time import strftime

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="auto")
    parser.add_argument("--gpu", type=str, default=None, help="Value for CUDA_VISIBLE_DEVICES, e.g. '0' or '1'.")
    parser.add_argument("--question-file", type=str, default=".data/ReVA_V2/valid_set.json")
    parser.add_argument("--dataset-root", type=str, default=".data")
    parser.add_argument("--dataset-prefix", type=str, default="#dataset")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-video-frames", type=int, default=-1)
    parser.add_argument("--video-max-tiles", type=int, default=-1)
    parser.add_argument("--generation-config", type=json.loads, default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_instances(question_file: str) -> list[dict]:
    with open(question_file) as f:
        data = json.load(f)

    instances = []
    for video_id, video_info in data["videos"].items():
        mcq = video_info.get("mcq", {})
        for category, subcategories in mcq.items():
            for subcategory, questions in subcategories.items():
                for question in questions:
                    instances.append(
                        {
                            "qa_id": question["qa_id"],
                            "video_id": video_id,
                            "video_path": video_info["file_path"],
                            "dataset_name": video_info.get("dataset_name"),
                            "category": category,
                            "subcategory": subcategory,
                            "question": question["question"],
                            "options": question["options"],
                            "correct_answer": question["correct_answer"],
                            "reasoning": question.get("reasoning", ""),
                            "example": question.get("example", ""),
                        }
                    )
    return instances


def resolve_video_path(raw_path: str, dataset_root: str, dataset_prefix: str) -> str:
    dataset_root_path = Path(dataset_root)
    candidates = [Path(raw_path), dataset_root_path / raw_path]

    if dataset_prefix in raw_path:
        suffix = raw_path.split(dataset_prefix, 1)[1].lstrip("/\\")
        suffix_path = Path(suffix)
        candidates.append(dataset_root_path / suffix_path)

        parts = suffix_path.parts
        if len(parts) > 1:
            candidates.append(dataset_root_path / Path(*parts[1:]))
            if dataset_root_path.exists():
                for child in dataset_root_path.iterdir():
                    if child.is_dir():
                        candidates.append(child / Path(*parts[1:]))

    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        resolved = str(candidate)
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists():
            return resolved

    raise FileNotFoundError(f"Cannot resolve video path '{raw_path}'. Tried: {', '.join(seen)}")


def build_prompt(question: str, options: dict[str, str]) -> str:
    labels = sorted(options.keys())
    lines = [question]
    lines.extend(f"{label}. {options[label]}" for label in labels)
    lines.append("Answer with only the option letter from the given choices.")
    return "\n".join(lines)


def parse_choice(response: str, options: dict[str, str]) -> str | None:
    response = response.strip()
    labels = sorted(options.keys())
    candidates = []

    padded = f" {response} "
    for label in labels:
        idx = padded.rfind(f"({label})")
        if idx != -1:
            candidates.append((idx, label))
    if not candidates:
        for label in labels:
            idx = padded.rfind(f" {label} ")
            if idx != -1:
                candidates.append((idx, label))
    if not candidates:
        for label in labels:
            if re.match(rf"^\s*{re.escape(label)}\s*[\.\:\)\]\u3001]?\s*$", response, flags=re.IGNORECASE):
                candidates.append((0, label))
    if not candidates:
        for label in labels:
            if re.match(rf"^\s*{re.escape(label)}\s*[\.\:\)\]\u3001]?\b", response, flags=re.IGNORECASE):
                candidates.append((0, label))
    if not candidates:
        lowered = response.lower()
        for label in labels:
            for pattern in [f"the answer is {label.lower()}", f"the letter is {label.lower()}"]:
                idx = lowered.rfind(pattern)
                if idx != -1:
                    candidates.append((idx, label))
    if not candidates:
        lowered = response.lower()
        for label in labels:
            option_text = options[label].strip().lower()
            idx = lowered.rfind(option_text)
            if idx != -1:
                candidates.append((idx, label))

    if not candidates:
        return None
    return max(candidates)[1]


def load_existing_predictions(output_path: str) -> dict[str, dict]:
    if not os.path.exists(output_path):
        return {}
    predictions = {}
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            predictions[record["qa_id"]] = record
    return predictions


def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_jsonl(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(records: list[dict]) -> dict:
    answered = [record for record in records if record.get("pred_letter") is not None]
    correct = sum(record["is_correct"] for record in records)

    metrics = {
        "num_questions": len(records),
        "num_answered": len(answered),
        "accuracy": correct / len(records) if records else 0.0,
    }

    by_category = {}
    by_subcategory = {}
    for record in records:
        for bucket, key in ((by_category, record["category"]), (by_subcategory, record["subcategory"])):
            stats = bucket.setdefault(key, {"count": 0, "correct": 0})
            stats["count"] += 1
            stats["correct"] += int(record["is_correct"])

    for bucket in (by_category, by_subcategory):
        for stats in bucket.values():
            stats["accuracy"] = stats["correct"] / stats["count"] if stats["count"] else 0.0

    metrics["by_category"] = by_category
    metrics["by_subcategory"] = by_subcategory
    return metrics


def main() -> None:
    args = parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import llava
    from llava import Video
    from llava import conversation as conversation_lib

    instances = load_instances(args.question_file)
    if args.max_questions is not None:
        instances = instances[: args.max_questions]

    base_output_dir = Path(args.output_dir)
    if args.resume:
        output_dir = base_output_dir
    else:
        output_dir = base_output_dir / strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving evaluation artifacts to: {output_dir}")

    outputs_path = output_dir / "outputs.jsonl"
    metrics_path = output_dir / "metrics.json"

    existing = load_existing_predictions(str(outputs_path)) if args.resume else {}

    if args.conv_mode != "auto":
        conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_mode].copy()

    model = llava.load(args.model_path, model_base=args.model_base)
    if args.num_video_frames > 0:
        model.config.num_video_frames = args.num_video_frames
    if args.video_max_tiles > 0:
        model.config.video_max_tiles = args.video_max_tiles
        model.llm.config.video_max_tiles = args.video_max_tiles

    generation_config = copy.deepcopy(model.default_generation_config)
    if args.generation_config is not None:
        generation_config.update(**args.generation_config)

    outputs = []
    for instance in tqdm(instances):
        if instance["qa_id"] in existing:
            outputs.append(existing[instance["qa_id"]])
            continue

        video_path = resolve_video_path(instance["video_path"], args.dataset_root, args.dataset_prefix)
        prompt = build_prompt(instance["question"], instance["options"])
        response = model.generate_content(
            [Video(video_path), prompt],
            generation_config=generation_config,
        )
        pred_letter = parse_choice(response, instance["options"])

        outputs.append(
            {
                "qa_id": instance["qa_id"],
                "video_id": instance["video_id"],
                "video_path": video_path,
                "dataset_name": instance["dataset_name"],
                "category": instance["category"],
                "subcategory": instance["subcategory"],
                "question": instance["question"],
                "options": instance["options"],
                "prompt": prompt,
                "raw_response": response,
                "pred_letter": pred_letter,
                "correct_answer": instance["correct_answer"],
                "is_correct": pred_letter == instance["correct_answer"],
                "reasoning": instance["reasoning"],
                "example": instance["example"],
            }
        )

        save_jsonl(str(outputs_path), outputs)

    outputs.sort(key=lambda record: record["qa_id"])
    save_jsonl(str(outputs_path), outputs)
    save_json(str(metrics_path), summarize(outputs))


if __name__ == "__main__":
    main()
