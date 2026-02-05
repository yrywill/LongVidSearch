import os
import sys
import json
import logging
import random
import re
import argparse
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq
import numpy as np
from openai import OpenAI
import ast
from typing import Any, Optional

from tools import search_clips_in_video,FINAL_ANSWER,get_clip_detail
from utils_general import get_from_cache, save_to_cache

def setup_logger(log_file: str):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == log_file for h in logger.handlers):
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [T:%(threadName)s|%(thread)d] - %(message)s (line %(lineno)d)"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def setup_client(base_url: str, api_key: str):
    if not api_key:
        raise ValueError("api_key is empty. Pass --api_key or set env OPENAI_API_KEY.")
    return OpenAI(base_url=base_url, api_key=api_key)


def parse_args():
    p = argparse.ArgumentParser(description="VideoAgent runner with argparse config")

    # ====== IO Paths ======
    p.add_argument("--input_ann_file", type=str, required=True,
                   help="Path to merged annotations json (anns).")
    p.add_argument("--all_cap_file", type=str, required=True,
                   help="Path to merged parquet captions file.")
    p.add_argument("--output_json", type=str, required=True,
                   help="Path to output logs json.")
    p.add_argument("--log_file", type=str, required=True,
                   help="Path to log file.")

    # ====== OpenAI compatible endpoint ======
    p.add_argument("--base_url", type=str, required=True,
                   help="OpenAI compatible base_url.")
    p.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY", ""),
                   help="API key. Default from env OPENAI_API_KEY.")
    p.add_argument("--model", type=str, required=True,
                   help="Model name for chat.completions.")

    # ====== Runtime ======
    p.add_argument("--max_workers", type=int, default=1,
                   help="ThreadPoolExecutor max_workers.")

    return p.parse_args()

def parse_json(text: str) -> Optional[Any]:
    if text is None:
        return None
    # --- normalize smart quotes  ---
    text = (
        text.replace("“", '\\"')
            .replace("”", '\\"')
            .replace("‘", "\\'")
            .replace("’", "\\'")
    ).strip()
    # ----------  strict JSON ----------
    try:
        return json.loads(text)
    except Exception:
        pass
    # ----------  Python literal  ----------
    try:
        return ast.literal_eval(text)
    except Exception:
        pass
    # ----------  Markdown code block ----------
    blocks = re.findall(
        r"```(?:json|python)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    for block in blocks:
        block = block.strip()
        try:
            return json.loads(block)
        except Exception:
            pass
        try:
            return ast.literal_eval(block)
        except Exception:
            pass

    # ----------  extract {} / [] ----------
    def extract_balanced(s: str, open_c: str, close_c: str):
        results = []
        stack = 0
        start = None
        in_str = False
        str_ch = None
        i = 0
        while i < len(s):
            ch = s[i]
            if in_str:
                if ch == "\\":
                    i += 2
                    continue
                if ch == str_ch:
                    in_str = False
                i += 1
                continue
            if ch in ("'", '"'):
                in_str = True
                str_ch = ch
                i += 1
                continue
            if ch == open_c:
                if stack == 0:
                    start = i
                stack += 1
            elif ch == close_c and stack > 0:
                stack -= 1
                if stack == 0 and start is not None:
                    results.append(s[start:i + 1])
                    start = None
            i += 1
        return results

    candidates = []
    candidates += extract_balanced(text, "{", "}")
    candidates += extract_balanced(text, "[", "]")
    candidates = sorted(set(candidates), key=len, reverse=True)
    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            pass
        try:
            return ast.literal_eval(cand)
        except Exception:
            continue
    return None

def parse_text_find_number(text):
    item = parse_json(text)
    try:
        match = int(item["final_answer"])
        if match in range(-1, 5):
            return match
        else:
            return random.randint(0, 4)
    except Exception as e:
        logger.error(f"Answer Parsing Error: {e}")
        return -1

def parse_text_find_confidence(text):
    item = parse_json(text)
    logger.info(f"Confidence Parsing Text: {item}")
    try:
        match = int(item["confidence"])
        if match in range(1, 4):
            return match
        else:
            return random.randint(1, 3)
    except Exception as e:
        logger.error(f"Confidence Parsing Error: {e}")
        return 1

def get_llm_response(
    system_prompt, prompt, json_schema,json_format=True):
    model = arg.model
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": prompt},
    ]
    key = json.dumps([model, messages])
    cached_value = get_from_cache(key)
    if cached_value is not None:
        logger.info("Cache Hit")
        logger.info(cached_value)
        return cached_value

    for _ in range(3):
        try:
            ###formetting###
            if json_format:
                completion = client.chat.completions.create(
                    model=model,
                    response_format={"type": "json_schema",
                        "json_schema": json_schema},
                    messages=messages,
                )
            else:
                completion = client.chat.completions.create(
                    model=model, messages=messages
                )
            response = completion.choices[0].message.content
            logger.info(response)
            save_to_cache(key, response)
            return response
        except Exception as e:
            logger.error(f"GPT Error: {e}")
            continue
    return "GPT Error"

def generate_final_answer(question, caption, num_frames):
    answer_format = {"final_answer": "xxx"}
    FINAL_ANSWER_SCHEMA = {
    "name": "final_answer_schema",
    "schema": {
        "type": "object",
        "properties": {
            "final_answer": {"type": "string"}
        },
        "required": ["final_answer"],
        "additionalProperties": False
    }
    }
    prompt = f"""
    You are given a video segmented into {num_frames} clips (each clip is approximately 30 seconds).
    The following are the clip-level captions, ordered chronologically:
    CAPTIONS:{caption}
    QUESTION:{question}
    INSTRUCTIONS:
    1) Answer the question using ONLY the information explicitly stated in the captions.
    2) If the captions provide clear and sufficient evidence, answer confidently and directly.
    3) If the evidence is partial, ambiguous, or insufficient, provide the best supported answer and mark uncertainty explicitly.
    4) If the question cannot be answered from the captions, state that the information is insufficient.
    OUTPUT FORMAT:
    Return a single JSON object that strictly matches the schema below:{answer_format}""".strip()

    system_prompt = """You are a video question-answering assistant operating in a multi-step reasoning system.
    Your goal is to provide a concise, factual answer based on the given video captions.
    Therefore, your answer should clearly reflect the strength of evidence available."""
    response = get_llm_response(system_prompt, prompt,FINAL_ANSWER_SCHEMA,json_format=True)
    return response

def generate_description_step(question, caption, num_frames, segment_des):
    formatted_description = {
        "clip_descriptions": [
            {"segment_id": "1", "duration": "xx - xx", "description": "clip of xx"},
            {"segment_id": "2", "duration": "xx - xx", "description": "clip of xx"},
            {"segment_id": "3", "duration": "xx - xx", "description": "clip of xx"},
            ...
        ]
    }
    CLIP_DESCRIPTIONS_SCHEMA = {
    "name": "clip_descriptions_schema",
    "schema": {
        "type": "object",
        "properties": {
        "clip_descriptions": {
            "type": "array",
            "minItems": 1,          
            "maxItems": 6,         ####### (exitable range 1-6)
            "items": {
            "type": "object",
            "properties": {
                "segment_id": {
                "type": "string",
                "pattern": "^\\d+$",   
                "description": "Segment id like '1', '2', ..."
                },
                "duration": {
                "type": "string",
                "description": "The segment range, e.g. '12-24' or 'xx - xx'."
                # "pattern": "^\\s*\\d+\\s*-\\s*\\d+\\s*$"
                },
                "description": {
                "type": "string",
                "minLength": 1,
                "description": "Concise description for the selected segment."
                }
            },
            "required": ["segment_id", "duration", "description"],
            "additionalProperties": False
            }
        }
        },
        "required": ["clip_descriptions"],
        "additionalProperties": False
    }
    }

    prompt = f"""
    Given a video that has {num_frames} clips, the clips are decoded at 30 seconds. Given the following descriptions of sampled clips in the video:
    {caption}
    To answer the following question: 
    ``` 
    {question}
    ``` 
    However, the information in the initial clips is not suffient.
    Objective:
    Our goal is to identify additional clips that contain crucial information necessary for answering the question. These clips should not only address the query directly but should also complement the insights gleaned from the descriptions of the initial clips.
    To achieve this, we will:
    1. Divide the video into segments based on the intervals between the initial clips as, candidate segments: {segment_des}
    2. Determine which segments are likely to contain clips that are most relevant to the question. These clips should capture key visual elements, such as objects, humans, interactions, actions, and scenes, that are supportive to answer the question.
    For each clip identified as potentially relevant, provide a concise description focusing on essential visual elements. Use a description per clip. If the specifics of a segment's visual content are uncertain based on the current information, use placeholders for specific actions or objects, but ensure the description still conveys the segment's relevance to the query.
    Select multiple clips from one segment if necessary to gather comprehensive insights. 
    Return the descriptions and the segment id in JSON format, note "segment_id" must be smaller than {len(segment_des) + 1}, "duration" should be the same as candiate segments:
    ```
    {formatted_description}
    ```
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    response = get_llm_response(system_prompt, prompt,CLIP_DESCRIPTIONS_SCHEMA, json_format=True)
    return response

def self_eval(num_frames,caption,question, answer):
    confidence_format = {"confidence": "xxx"}
    CONFIDENCE_SCHEMA = {
        "name": "confidence_schema",
        "schema": {
            "type": "object",
            "properties": {
                "confidence": {"type": "string", "enum": ["1", "2", "3"]}
            },
            "required": ["confidence"],
            "additionalProperties": False
        }
    }

    prompt = f"""Please assess the confidence level in the answering process.
    The provided information is as as follows：
    You are given a video segmented into {num_frames} clips (each clip is approximately 30 seconds).
    The following are the clip-level captions, ordered chronologically:
    CAPTIONS:{caption}
    QUESTION:{question}.
    The answering making process is as follows,
    {answer}
    Criteria for Evaluation:
    Insufficient Information (Confidence Level: 1): If information is too lacking for a reasonable conclusion.
    Partial Information (Confidence Level: 2): If information partially supports an informed guess.
    Sufficient Information (Confidence Level: 3): If information fully supports a well-informed decision.
    Assessment Focus:
    Evaluate based on the relevance, completeness, and clarity of the provided information in relation to the answering context.
    Please generate the confidence with JSON format {confidence_format}
    """
    system_prompt = "You are a helpful assistant designed to output JSON."
    response = get_llm_response(system_prompt, prompt,CONFIDENCE_SCHEMA, json_format=True)
    return response

def ask_gpt_caption(question, caption, num_frames):
    answer_format = {"final_answer": "xxx"}
    FINAL_ANSWER_SCHEMA = {
    "name": "final_answer_schema",
    "schema": {
        "type": "object",
        "properties": {
        "final_answer": {
            "type": "string",
            "description": "Final answer text. If insufficient, use 'insufficient information'."
        }
        },
        "required": ["final_answer"],
        "additionalProperties": False
    }
    }

    prompt = f"""
    You are given a video segmented into {num_frames} clips (each clip is approximately 30 seconds).
    The following are the clip-level captions, ordered chronologically:
    CAPTIONS:{caption}
    QUESTION:{question}
    INSTRUCTIONS:
    1) Answer the question using ONLY the information explicitly stated in the captions.
    2) If the captions provide clear and sufficient evidence, answer confidently and directly.
    3) If the evidence is partial, ambiguous, or insufficient, provide the best supported answer and mark uncertainty explicitly.
    4) If the question cannot be answered from the captions, state that the information is insufficient.
    OUTPUT FORMAT:
    Return a single JSON object that strictly matches the schema below:{answer_format}""".strip()

    system_prompt = """You are a video question-answering assistant operating in a multi-step reasoning system.
    Your goal is to provide a concise, factual answer based on the given video captions.
    Therefore, your answer should clearly reflect the strength of evidence available."""
    response = get_llm_response(system_prompt, prompt,FINAL_ANSWER_SCHEMA, json_format=False)
    return prompt, response

def ask_gpt_caption_step(question, caption, num_frames):
    answer_format = {"final_answer": "xxx"}
    FINAL_ANSWER_SCHEMA = {
    "name": "final_answer_schema",
    "schema": {
        "type": "object",
        "properties": {
        "final_answer": {
            "type": "string",
            "description": "Final answer text. If insufficient, use 'insufficient information'."
        }
        },
        "required": ["final_answer"],
        "additionalProperties": False
    }
    }
    prompt = f"""
    You are given a video segmented into {num_frames} clips (each clip is approximately 30 seconds).
    The following are the clip-level captions, ordered chronologically:
    CAPTIONS:{caption}
    QUESTION:{question}
    INSTRUCTIONS:
    1) Answer the question using ONLY the information explicitly stated in the captions.
    2) If the captions provide clear and sufficient evidence, answer confidently and directly.
    3) If the evidence is partial, ambiguous, or insufficient, provide the best supported answer and mark uncertainty explicitly.
    4) If the question cannot be answered from the captions, state that the information is insufficient.
    OUTPUT FORMAT:
    Return a single JSON object that strictly matches the schema below:{answer_format}""".strip()
    system_prompt = """You are a video question-answering assistant operating in a multi-step reasoning system.
    Your goal is to provide a concise, factual answer based on the given video captions.
    Therefore, your answer should clearly reflect the strength of evidence available."""
    response = get_llm_response(system_prompt, prompt,FINAL_ANSWER_SCHEMA, json_format=False)
    return prompt, response

def extract_caption(vid,all_cap_file):
    VID = vid
    PARQUET_PATH = all_cap_file
    table = pq.read_table(
        PARQUET_PATH,
        filters=[("vid", "==", VID)],
        columns=["vid", "slice_num", "cap"]
    )
    caps = [
        r["cap"]
        for r in sorted(table.to_pylist(), key=lambda x: x["slice_num"])
    ]
    return caps

def run_one_question(idx, ann, caps, logs):
    global_cost = 0
    logger.info(f"[##### {idx} starting #####]")
    question = ann["question"]
    vid=ann["vid"]
    formatted_question = (
        f"Here is the question: {question}"
    )
    num_frames = len(caps)
    ### Step 1 ###
    sample_idx = np.linspace(1, num_frames, num=5, dtype=int).tolist()
    
    sampled_caps = get_clip_detail(caps, sample_idx)
    previous_prompt, answer_str = ask_gpt_caption(
        formatted_question, sampled_caps, num_frames
    )
    confidence_str = self_eval(num_frames, sampled_caps, formatted_question, answer_str)
    confidence = parse_text_find_confidence(confidence_str)
    logger.info(f"#{idx} Step 1: confidence {confidence}")
    ### Step 2 ### 
    if confidence < 3:
        try:
            #separate segments
            segment_des = {
                i + 1: f"{sample_idx[i]}-{sample_idx[i + 1]}"
                for i in range(len(sample_idx) - 1)
            }
            candiate_descriptions = generate_description_step(
                formatted_question,
                sampled_caps,
                num_frames,
                segment_des,
            )
            parsed_candiate_descriptions = parse_json(candiate_descriptions)
            ####  query-->idx
            frame_idx = search_clips_in_video(
                parsed_candiate_descriptions["clip_descriptions"], vid, sample_idx,caps
            )
            logger.info(f"#{idx} Step 2: {frame_idx}")
            global_cost += 1
            sample_idx += frame_idx
            
            sample_idx = sorted(list(set(sample_idx)))

            sampled_caps = get_clip_detail(caps, sample_idx)
            previous_prompt, answer_str = ask_gpt_caption_step(
                formatted_question, sampled_caps, num_frames
            )
            
            confidence_str = self_eval(num_frames, sampled_caps, formatted_question, answer_str)
            confidence = parse_text_find_confidence(confidence_str)
            logger.info(f"#{idx} Step 2: confidence {confidence}")
        except Exception as e:
            logger.error(f"Step 2 Error: {e}")
            answer_str = generate_final_answer(
                formatted_question, sampled_caps, num_frames
            )
            

    ### Step 3 ###
    if confidence < 3:
        try:
            segment_des = {
                i + 1: f"{sample_idx[i]}-{sample_idx[i + 1]}"
                for i in range(len(sample_idx) - 1)
            }
            candiate_descriptions = generate_description_step(
                formatted_question,
                sampled_caps,
                num_frames,
                segment_des,
            )
            parsed_candiate_descriptions = parse_json(candiate_descriptions)
            frame_idx = search_clips_in_video(
                parsed_candiate_descriptions["clip_descriptions"], vid, sample_idx, caps
            )
            logger.info(f"#{idx} Step 3: {frame_idx}")
            global_cost += 1
            sample_idx += frame_idx
            sample_idx = sorted(list(set(sample_idx)))
            sampled_caps = get_clip_detail(caps, sample_idx)
            answer_str = generate_final_answer(
                formatted_question, sampled_caps, num_frames
            )
            #answer = parse_text_find_number(answer_str)
        except Exception as e:
            logger.error(f"#{idx} Step 3 Error: {e}")
            answer_str = generate_final_answer(
                formatted_question, sampled_caps, num_frames
            )
            #answer = parse_text_find_number(answer_str)
    logger.info(f"#{idx} Final Answer String: {answer_str}")
    parsed = parse_json(answer_str)
    answer = parsed.get("final_answer") if isinstance(parsed, dict) else None
   
    if not answer :
        logger.error(f"#{idx} Answer Not Found!")
    else:
        logger.info(f"Finished video:{idx}/{vid}/{answer}/{ann['answer']}")

    corr_answer = ann["answer"]
    golden_evidence_clip = ann["evidence_slices"]
    category = ann["category"]
    hop_level = ann["hop_level"]
    #count_frame = len(sample_idx)
    reasoning_chain = ann["reasoning_chain"]
    corr1,corr2,corr3,cost = FINAL_ANSWER(question,answer,corr_answer,reasoning_chain,global_cost)
    if corr1+corr2+corr3 >=2:
        general_corr = 1
    else:
        general_corr = 0
    logger.info(f"### {idx} QA Log ###: \n#question: {question}, \n#answer: {answer}, \n#corr_answer: {corr_answer}, \n#general_corr: {general_corr}, \n#corr1: {corr1}, \n#corr2: {corr2}, \n#corr3: {corr3}, \n#cost: {cost}, \n#golden_evidence_clip: {golden_evidence_clip}, \n#sampled_clips: {sample_idx}")
    
    logs[idx] = {
        "question": question,
        "answer": answer,
        "corr_answer": corr_answer,
        "general_corr": general_corr,
        "corr1": corr1,
        "corr2": corr2,
        "corr3": corr3,
        "cost": cost,
        "category": category,
        "hop_level": hop_level,
        "golden_evidence_clip": golden_evidence_clip,
        "sampled_clips": sample_idx
    }
    
    

def main(args):
    # if running full set, change subset to fullset
    
    input_ann_file = args.input_ann_file
    all_cap_file = args.all_cap_file
    json_file_name = args.output_json
    
    anns = json.load(open(input_ann_file, "r", encoding="utf-8"))
    logs = {}
    
    tasks = [
        (ann, extract_caption(ann["vid"], all_cap_file), logs)
        for ann in anns]
    for i, task in enumerate(tasks):
        ann, caption, logs = task
        if caption is None:
            print(f"[{i}] vid={ann['vid']} -> caption is None")
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        executor.map(lambda idx_task: run_one_question(idx_task[0], *idx_task[1]), enumerate(tasks))
                     
    def json_default(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.ndarray,)): return o.tolist()
        raise TypeError(f"Not JSON serializable: {type(o)}")
    total = len(logs)
    accuracy1 = sum(v["corr1"] for v in logs.values()) / total if total > 0 else 0.0
    accuracy2 = sum(v["corr2"] for v in logs.values()) / total if total > 0 else 0.0
    accuracy3 = sum(v["corr3"] for v in logs.values()) / total if total > 0 else 0.0
    accuracy = sum(v["general_corr"] for v in logs.values()) / total if total > 0 else 0.0
    logger.info(f"Accuracy: {accuracy1:.4f},accurate : {sum(v['corr1'] for v in logs.values())} / {total} ")
    logger.info(f"Accuracy: {accuracy2:.4f},accurate : {sum(v['corr2'] for v in logs.values())} / {total} ")
    logger.info(f"Accuracy: {accuracy3:.4f},accurate : {sum(v['corr3'] for v in logs.values())} / {total} ")
    logger.info(f"general Accuracy: {accuracy:.4f},accurate : {sum(v['general_corr'] for v in logs.values())} / {total} ")
    logs = {k: logs[k] for k in sorted(logs.keys(), key=lambda x: int(x))}
    count=0
    for idx, v in logs.items():
        corr1 = v.get("corr1")
        corr2 = v.get("corr2")
        corr3 = v.get("corr3")
        if corr1 != corr2 or corr1 != corr3 or corr2 != corr3:
            logger.info(f"[difference] idx={idx} corr1={corr1} corr2={corr2} corr3={corr3}")
            count+=1
    logger.info(f"Total different answers count: {count}")
    with open(json_file_name, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2, default=json_default)

if __name__ == "__main__":
    arg = parse_args()
    logger = setup_logger(arg.log_file)
    client = setup_client(arg.base_url, arg.api_key)
    main(arg)