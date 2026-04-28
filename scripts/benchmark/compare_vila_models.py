import argparse
import copy
import gc
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

STRICT_BENCHMARK_GENERATION_CONFIG = {
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "num_beams": 1,
    "max_new_tokens": 4,
}


def get_cli_generation_config(override: dict | None) -> dict[str, Any]:
    config = dict(STRICT_BENCHMARK_GENERATION_CONFIG)
    if override is not None:
        config.update(override)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a-path", type=str, required=True)
    parser.add_argument("--model-b-path", type=str, required=True)
    parser.add_argument("--model-a-name", type=str, default="model_a")
    parser.add_argument("--model-b-name", type=str, default="model_b")
    parser.add_argument("--gpu", type=str, default=None, help="Value for CUDA_VISIBLE_DEVICES, e.g. '0' or '1'.")
    parser.add_argument("--media", type=str, required=True, help="Path to one input media file.")
    parser.add_argument("--text", type=str, required=True, help="Prompt text used for both models.")
    parser.add_argument("--conv-mode", type=str, default="auto")
    parser.add_argument("--num-video-frames", type=int, default=-1)
    parser.add_argument("--video-max-tiles", type=int, default=-1)
    parser.add_argument("--generation-config", type=json.loads, default=None)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--benchmark-runs", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="runs/benchmark")
    return parser.parse_args()


def synchronize_if_possible() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


class StageTimer:
    def __init__(self) -> None:
        self.timings = defaultdict(float)

    def wrap_function(self, name: str, fn):
        def wrapped(*args, **kwargs):
            synchronize_if_possible()
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            synchronize_if_possible()
            self.timings[name] += time.perf_counter() - t0
            return result

        return wrapped

    def snapshot(self) -> dict[str, float]:
        return dict(self.timings)


def install_timing_hooks(model, timer: StageTimer):
    import llava.model.llava_arch as llava_arch_mod

    originals = {
        "extract_media": llava_arch_mod.extract_media,
        "process_image": llava_arch_mod.process_image,
        "process_images": llava_arch_mod.process_images,
        "tokenize_conversation": llava_arch_mod.tokenize_conversation,
        "generate": model.generate,
        "decode": model.tokenizer.decode,
    }

    llava_arch_mod.extract_media = timer.wrap_function("extract_media_s", llava_arch_mod.extract_media)
    llava_arch_mod.process_image = timer.wrap_function("vision_preprocess_s", llava_arch_mod.process_image)
    llava_arch_mod.process_images = timer.wrap_function("vision_preprocess_s", llava_arch_mod.process_images)
    llava_arch_mod.tokenize_conversation = timer.wrap_function("tokenize_s", llava_arch_mod.tokenize_conversation)
    model.generate = timer.wrap_function("generate_s", model.generate)
    model.tokenizer.decode = timer.wrap_function("decode_s", model.tokenizer.decode)

    return originals


def restore_timing_hooks(model, originals: dict[str, Any]) -> None:
    import llava.model.llava_arch as llava_arch_mod

    llava_arch_mod.extract_media = originals["extract_media"]
    llava_arch_mod.process_image = originals["process_image"]
    llava_arch_mod.process_images = originals["process_images"]
    llava_arch_mod.tokenize_conversation = originals["tokenize_conversation"]
    model.generate = originals["generate"]
    model.tokenizer.decode = originals["decode"]


def build_prompt_parts(media_path: str, text: str):
    import cv2
    from llava.media import Image, Video

    prompt = []
    lower = media_path.lower()
    if lower.endswith((".jpg", ".jpeg", ".png")):
        prompt.append(Image(media_path))
    elif lower.endswith((".mp4", ".mkv", ".webm")):
        cap = cv2.VideoCapture(media_path)
        if cap.isOpened():
            cap.release()
        prompt.append(Video(media_path))
    else:
        raise ValueError(f"Unsupported media type: {media_path}")
    prompt.append(text)
    return prompt


def benchmark_model(
    *,
    model_path: str,
    model_name: str,
    media_path: str,
    text: str,
    conv_mode: str,
    num_video_frames: int,
    video_max_tiles: int,
    generation_config_override: dict | None,
    warmup_runs: int,
    benchmark_runs: int,
) -> dict[str, Any]:
    import llava
    import torch
    from llava import conversation as clib

    model = None
    try:
        synchronize_if_possible()
        t0 = time.perf_counter()
        model = llava.load(model_path, model_base=None)
        synchronize_if_possible()
        load_model_s = time.perf_counter() - t0

        if num_video_frames > 0:
            model.config.num_video_frames = num_video_frames
        if video_max_tiles > 0:
            model.config.video_max_tiles = video_max_tiles
            model.llm.config.video_max_tiles = video_max_tiles

        if conv_mode != "auto":
            clib.default_conversation = clib.conv_templates[conv_mode].copy()

        generation_config = copy.deepcopy(model.default_generation_config)
        generation_config.update(**STRICT_BENCHMARK_GENERATION_CONFIG)
        if generation_config_override is not None:
            generation_config.update(**generation_config_override)

        prompt = build_prompt_parts(media_path, text)

        warmup_outputs = []
        for _ in range(warmup_runs):
            _ = model.generate_content(copy.deepcopy(prompt), generation_config=generation_config)

        runs = []
        for _ in range(benchmark_runs):
            timer = StageTimer()
            originals = install_timing_hooks(model, timer)
            try:
                synchronize_if_possible()
                t0 = time.perf_counter()
                response = model.generate_content(copy.deepcopy(prompt), generation_config=generation_config)
                synchronize_if_possible()
                total_infer_s = time.perf_counter() - t0
            finally:
                restore_timing_hooks(model, originals)

            prompt_token_count = len(model.tokenizer(text).input_ids)
            output_token_count = len(model.tokenizer.encode(response, add_special_tokens=False))
            run = {
                "total_infer_s": total_infer_s,
                "prompt_token_count": prompt_token_count,
                "output_token_count": output_token_count,
                "response_preview": response[:300],
            }
            run.update(timer.snapshot())
            runs.append(run)
            warmup_outputs.append(response)

        numeric_keys = sorted({k for run in runs for k, v in run.items() if isinstance(v, (int, float))})
        averages = {}
        for key in numeric_keys:
            averages[key] = sum(run.get(key, 0.0) for run in runs) / len(runs)

        return {
            "model_name": model_name,
            "model_path": model_path,
            "load_model_s": load_model_s,
            "effective_generation_config": generation_config.to_dict(),
            "warmup_runs": warmup_runs,
            "benchmark_runs": benchmark_runs,
            "runs": runs,
            "averages": averages,
        }
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            synchronize_if_possible()


def main() -> None:
    args = parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    output_dir = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    common = {
        "media_path": args.media,
        "text": args.text,
        "conv_mode": args.conv_mode,
        "num_video_frames": args.num_video_frames,
        "video_max_tiles": args.video_max_tiles,
        "generation_config_override": args.generation_config,
        "warmup_runs": args.warmup_runs,
        "benchmark_runs": args.benchmark_runs,
    }

    print(f"Saving benchmark artifacts to: {output_dir}")
    print(f"Media: {args.media}")
    print(f"Prompt: {args.text}")
    print(f"Benchmark generation config: {json.dumps(get_cli_generation_config(args.generation_config), ensure_ascii=False)}")

    result_a = benchmark_model(
        model_path=args.model_a_path,
        model_name=args.model_a_name,
        **common,
    )
    result_b = benchmark_model(
        model_path=args.model_b_path,
        model_name=args.model_b_name,
        **common,
    )

    report = {
        "config": {
            "gpu": args.gpu,
            "media": args.media,
            "text": args.text,
            "conv_mode": args.conv_mode,
            "num_video_frames": args.num_video_frames,
            "video_max_tiles": args.video_max_tiles,
            "generation_config": get_cli_generation_config(args.generation_config),
            "warmup_runs": args.warmup_runs,
            "benchmark_runs": args.benchmark_runs,
        },
        "models": [result_a, result_b],
    }

    with (output_dir / "report.json").open("w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print()
    for result in (result_a, result_b):
        avg = result["averages"]
        print(f"[{result['model_name']}]")
        print(f"  load_model_s:       {result['load_model_s']:.4f}")
        print(f"  avg total_infer_s:  {avg.get('total_infer_s', 0.0):.4f}")
        print(f"  avg extract_media:  {avg.get('extract_media_s', 0.0):.4f}")
        print(f"  avg preprocess:     {avg.get('vision_preprocess_s', 0.0):.4f}")
        print(f"  avg tokenize:       {avg.get('tokenize_s', 0.0):.4f}")
        print(f"  avg generate:       {avg.get('generate_s', 0.0):.4f}")
        print(f"  avg decode:         {avg.get('decode_s', 0.0):.4f}")
        print(f"  avg output_tokens:  {avg.get('output_token_count', 0.0):.2f}")
        print()


if __name__ == "__main__":
    main()
