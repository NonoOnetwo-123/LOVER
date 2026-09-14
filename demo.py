import argparse
import torch
from transformers import AutoProcessor # Assuming you use HuggingFace for model/processor loading
import os
import re
from src.vllm_inference.vllm_infer import vllmWrapper # Core inference logic
from src.vllm_inference.utils import _read_video_decord_w_timestamp, monkey_patch # Video processing
from src.utils.vision_process import smart_nframes # Video processing helper
from src.utils import process_vision_info_v3
import time
import json
# Apply monkey patch for video reading if necessary
monkey_patch()

PROMPT_TEMPLATE = """
You are given a video and a multiple-choice question.
Each video frame has a timestamp (shown as numbers on the screen) indicating the exact time of that scene. Use these timestamps to locate the relevant moments that answer the question.
Question:{}
Task:
1. First, locate the time period in the video that is most relevant to answering the question.
2. Then, select the correct answer from A, B, C, or D.
Output format:
- Output your thought process within the <think> </think> tags, including analysis with either specific time ranges (xx.xx to xx.xx) in <timestep> </timestep> tags.
- Then, provide the start and end times (in seconds, precise to two decimal places) in the format "start time to end time" within the <answer> </answer> tags. 
- Put your final choice inside <choice> </choice>.
Example:
<think>… </think>
<answer>12.54 to 17.83</answer>
<choice>B</choice>
"""

def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluation for training-free video temporal grounding (Single GPU Version)"
    )
    parser.add_argument(
        "--model_base", type=str, default="./ckpts/LOVER-7B"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="logs/",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:1", help="GPU device to use"
    )
    parser.add_argument(
        "--pipeline_parallel_size", type=int, default=1, help="GPU nodes"
    )
    parser.add_argument(
        "--video_path", type=str, default=""
    )
    parser.add_argument(
        "--query", type=str, default=""
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--total_pixels", type=int, default=3584 * 28 * 28, help="total_pixels"
    )
    return parser.parse_args()


def preprocess(processor, itm, ele):
    if "video_start" in itm and itm["video_start"] is not None:
        ele["video_start"] = itm["video_start"]
    if "video_end" in itm and itm["video_end"] is not None:
        ele["video_end"] = itm["video_end"]
    ele["nframes"] = 128  # 添加这一行，设置全视频均匀采样128帧

    messages = [
        {"role": "system", "content": []},
        {"role": "user", "content": []},
    ]
    messages[0]["content"].append({"type": "text", "text": "You are a helpful assistant."})
    messages[1]["content"].append({"type": "video", "video": itm["video"], **ele})
    messages[1]["content"].append(
        {
            "type": "text",
            "text": PROMPT_TEMPLATE.format(itm["sentence"]),
        }
    )
    _, video_inputs, utils = process_vision_info_v3(
        messages, return_video_kwargs=True
    )

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


    return {"text": text, "videos": video_inputs, "fps": utils["fps"]}


def build_dataset(
    data,
    processor,
    num_workers=8,
    sys_prompt="You are a helpful assistant.",
    min_pixels=16 * 28 * 28,
    total_pixels=3584 * 28 * 28,
    use_huggingface=False,
):
    kwargs = {
        "min_pixels": min_pixels,
        "total_pixels": total_pixels,
        "sys_prompt": sys_prompt,
    }
    ele = {
        "min_pixels": min_pixels,
        "total_pixels": total_pixels,
    }
    inputs = preprocess(processor, data, ele)

    multi_modal_data = {}
    if "images" in inputs and inputs["images"] is not None:
        multi_modal_data["image"] = inputs["images"]
    if "videos" in inputs and inputs["videos"] is not None:
        multi_modal_data["video"] = inputs["videos"]

    return {
        "inputs": {
            "raw_prompt_ids": [processor.tokenizer.encode(
                inputs["text"], add_special_tokens=False
            )],
            "multi_modal_data": [multi_modal_data],
            "mm_processor_kwargs": [(
                {"fps": inputs["fps"]} if inputs["fps"] is not None else {}
            )],
        },
        "timestamps": [data["timestamp"]],
        "duration": [data["duration"]],
        "video_paths": [data["video"]],
    }


def extract_answer(output_string):
    matches = re.findall(r"(\d+\.?\d*) (to|and) (\d+\.?\d*)", output_string)
    if not matches:
        answer_match = re.search(r"<answer>(.*?)</answer>", output_string)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            answer_matches = re.findall(
                r"(\d+\.?\d*) (to|and) (\d+\.?\d*)", answer_content
            )
            if answer_matches:
                last_match = answer_matches[-1]
                return [float(last_match[0]), float(last_match[2])]
        return [None, None]

    last_match = matches[-1]
    start_time_str = last_match[0]
    end_time_str = last_match[2]

    try:
        start_time = float(start_time_str)
        end_time = float(end_time_str)
        return [start_time, end_time]
    except ValueError:
        return [None, None]

def extract_choice(output_string):
    choice_match = re.findall(
        r"<choice>\s*([ABCDEFGHIJK])\s*</choice>",
        output_string,
        re.IGNORECASE
    )
    if choice_match:
        return choice_match[-1].upper()
    return None

def main(args):
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir, f"tmp_output_data.jsonl"
    )
    # build model and processor
    processor = AutoProcessor.from_pretrained(args.model_base, use_fast=True,  local_files_only=True)
    processor.tokenizer.padding_side = "left"
    model = vllmWrapper(args)

    data = {
        "video": args.video_path,
        "duration": 35.04, # Duration of the video in seconds, read the whole video
        "timestamp": [ # GT timestamps
            1.0,
            7.5
        ],
        "sentence": args.query,
    }

    data_args = {
        "num_workers": min(8, args.batch_size),
        "total_pixels": args.total_pixels,
    }
    data = build_dataset(data, processor, **data_args)

    program_start_time = time.perf_counter()

    output_texts = model.generate(
        data["inputs"],
        max_new_tokens=args.max_new_tokens,
    )
    targets = data["timestamps"]
    f = open(output_file, "a+")

    for i in range(len(targets)):
        pred = extract_answer(output_texts[i])
        pred_choice = extract_choice(output_texts[i])
        print(output_texts[i], pred)
        print("Pred time:", pred, "Pred choice:", pred_choice)
        f.write(
            json.dumps(
                {
                    "pred": pred,
                    "pred_choice": pred_choice,
                    "target": list(targets[i]),
                    "duration": (
                        None
                        if "duration" not in data
                        else data["duration"][i]
                    ),
                    "output_text": output_texts[i],
                }
            )
            + "\n"
        )
        f.flush()


    # --- END TOTAL TIME & CALCULATIONS ---
    program_end_time = time.perf_counter()
    total_program_duration = program_end_time - program_start_time

    print("\n--- Timing Summary ---")
    print(f"Total program execution time: {total_program_duration:.2f} seconds")

    output_filename = f"{args.output_dir}/timing_summary_vllm_data.txt"

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("\n--- Timing Summary ---\n")
        f.write(f"Total program execution time: {total_program_duration:.2f} seconds\n")
        f.write("Another line of summary using write.\n")

    



if __name__ == "__main__":
    from src.vllm_inference.utils import monkey_patch

    monkey_patch()
    args = get_args()
    main(args)