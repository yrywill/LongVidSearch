import os
import re
import ast
import json
from openai import OpenAI
from typing import List
import numpy as np
import logging

logger = logging.getLogger(__name__)

##### Both need to add api_key and base_url first ######
client = OpenAI(
      base_url="",
      api_key="")

embed = OpenAI(
    base_url="", 
    api_key="")

### Tool 1 ###
def get_clip_detail(captions, sample_idx):
    video_caption = {}
    for idx in sample_idx:
        video_caption[f"frame {idx}"] = captions[idx - 1]
    return video_caption

def embed_texts(
    texts,
    embed_model ="",
    batch_size=1,
    vllm_available=True,
):
    """
    Args:
        texts (List[str]): 
        embed_model (str): 
            - "vllm:xxx" use vLLM embedding
            - others SentenceTransformer
        batch_size (int): SentenceTransformer batch size
        vllm_available (bool):  

    Returns:
        np.ndarray: shape (N, D), float32, L2-normalized
    """
    # ----------- vLLM embedding -----------
    if vllm_available:
        from vllm import LLM
        model_name = embed_model
        logger.info(f"[EMBED] vLLM model: {model_name}")
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            task="embed"
        )
        outputs = llm.embed(texts)
        # vLLM 
        embs = np.array(
            [o.outputs.embedding for o in outputs],
            dtype=np.float32
        )
    # -------- SentenceTransformer embedding --------
    else:
        from sentence_transformers import SentenceTransformer

        logger.info(f"[EMBED] SentenceTransformer: {embed_model}")

        model = SentenceTransformer(embed_model)
        embs = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
        ).astype(np.float32)
    return np.ascontiguousarray(embs)

def get_embeddings(inputs: List[str]) -> np.ndarray:
    resp = embed.embeddings.create(
        model="qwen3-embedding-0.6B",######Need to change to your embedding model######
        input=inputs,
        encoding_format="float",
        )
    embeddings = [d.embedding for d in resp.data]
    return np.array(embeddings)

def get_batch_embeddings(
        
    inputs: List[str],
    embed_model: str = "/mnt/DataFlow/yry/model/qwen3-embedding-0.6B",
    batch_size: int = 32,
    vllm_available: bool = True,
) -> np.ndarray:
    """
    Compute batch embeddings for a list of texts.

    Args:
        inputs (List[str]): input texts
        embed_model (str): embedding model name or path
        batch_size (int): batch size for SentenceTransformer
        vllm_available (bool): whether to use vLLM backend

    Returns:
        np.ndarray: shape (N, D), dtype float32
    """
    assert isinstance(inputs, list) and len(inputs) > 0
    logger.info(f"[EMBED] Num inputs: {len(inputs)}")
    # ----------- vLLM embedding -----------
    if vllm_available:
        from vllm import LLM
        logger.info(f"[EMBED] Using vLLM model: {embed_model}")
        llm = LLM(
            model=embed_model,
            trust_remote_code=True,
            task="embed",
        )
        # vLLM embed batch
        outputs = llm.embed(inputs)
        # outputs: List[RequestOutput]
        embs = np.asarray(
            [o.outputs.embedding for o in outputs],
            dtype=np.float32,
        )
    # -------- SentenceTransformer embedding --------
    else:
        from sentence_transformers import SentenceTransformer
        logger.info(f"[EMBED] Using SentenceTransformer: {embed_model}")
        model = SentenceTransformer(embed_model)
        embs = model.encode(
            inputs,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)
    return embs

### Tool 2 ###
def search_clips_in_video(descriptions, video_id, sample_idx,caps):
    if not os.path.exists(f"./video_embeddings/frame_embeddings_{video_id}.npy"):
        frame_embeddings = get_embeddings(caps)
        print(frame_embeddings.shape)
        np.save(f"./video_embeddings/frame_embeddings_{video_id}.npy", frame_embeddings)   
    else:
        frame_embeddings = np.load(f"./video_embeddings/frame_embeddings_{video_id}.npy")

    text_embedding = get_embeddings(
        [description["description"] for description in descriptions]
    )
    frame_idx = []
    for idx, description in enumerate(descriptions):
        seg = int(description["segment_id"]) - 1
        seg_frame_embeddings = frame_embeddings[sample_idx[seg] : sample_idx[seg + 1]]
        if seg_frame_embeddings.shape[0] < 2:
            frame_idx.append(sample_idx[seg] + 1)
            continue
        seg_similarity = text_embedding[idx] @ seg_frame_embeddings.T
        seg_frame_idx = sample_idx[seg] + seg_similarity.argmax() + 1
        frame_idx.append(seg_frame_idx)

    return frame_idx

def get_llm_response(
    system_prompt, prompt,json_schema, json_format=True, model= None
):
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": prompt},
    ]
    
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
            return response
        except Exception as e:
            logger.error(f"GPT Error: {e}")
            continue
    return "GPT Error"

def parse_json(text: str):
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

    # ----------  extract {} / [] ---------
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
    
def parse_bool_from_json(response_json):

    parsed_json = parse_json(response_json)
    if parsed_json and "judgement_result" in parsed_json:
        return parsed_json["judgement_result"]
    else:
        raise ValueError("Invalid JSON format or 'judgement_result' key not found.")
    
### Tool 3 ###  
def FINAL_ANSWER(question,answer,corr_answer,reasoning_chain,cost):
    prompt = f"""
    You are an evaluator judging whether an answer should be considered correct.

    IMPORTANT CONTEXT:
    - Do NOT assume or infer any video details beyond what is explicitly in the Question, the Reference Answer, and the Reasoning Chain.
    - The Reference Answer is the ground-truth semantic target. Judge semantic consistency with it.
    - If the Question and the Reference Answer appear inconsistent, follow the Reference Answer.

    INPUTS:
    Question: {question}
    Reference Answer (ground truth): {corr_answer}
    Reasoning Chain (gold rationale, auxiliary): {reasoning_chain}
    Answer to Evaluate: {answer}

    EVALUATION GOAL:
    Return judgement_result = true iff the Answer to Evaluate:
    1) is addressing the same target as the Question, AND
    2) is semantically consistent with the Reference Answer, preserving all REQUIRED information.

    PROCEDURE:
    1) Extract REQUIRED KEY POINTS (1~5 atomic points) from the Reference Answer in the context of the Question.
    - You MAY use the Reasoning Chain only to clarify/disambiguate what the Reference Answer means (which hop/entity it refers to).
    - You MUST NOT add new REQUIRED points that go beyond what the Reference Answer implies.
    - Extra specifics appearing only in the Reasoning Chain are OPTIONAL, not required.
    - If Reasoning Chain conflicts with Reference Answer, follow Reference Answer.
    2) Check whether the Answer to Evaluate covers each required key point (paraphrase allowed).
    3) Decide true/false using the rules below.

    RULES:
    A. REQUIRED COVERAGE (STRICT)
    - The evaluated answer must include ALL required key points (paraphrase allowed).
    - If any required key point is missing or wrong -> false.
    - If the reference is specific (names/numbers/titles/locations), the evaluated answer must match that specificity;
    being more vague counts as missing -> false.

    A0. REFERENCE-SATISFIED PRIORITY (SOFT)
    - If ALL required key points from the Reference Answer are present in the evaluated answer
    (paraphrase allowed, and specificity is not reduced),
    then prefer judgement_result = true,
    unless it triggers a clear contradiction under Rule B
    or introduces a high-risk unsupported specific claim under Rule C.

    B. NO CONTRADICTIONS (REFERENCE-FIRST)
    - If the evaluated answer contradicts any required key point -> false.
    - Only enforce Question constraints when they are clearly compatible with the Reference.
    If a Question constraint conflicts with the Reference, ignore that constraint.

    C. RISKY UNSUPPORTED SPECIFICS (ONLY THESE CAUSE PENALTY)
    - Extra details are generally allowed.
    - Mark false ONLY if the evaluated answer introduces a high-risk specific claim NOT supported by Question/Reference/Reasoning Chain
    that could easily be wrong and would change meaning, such as:
    exact identity/name, exact count/number, exact brand, exact attacker/cause, exact named location/time.
    - Hedged speculation words like "maybe/likely/possibly" are ALLOWED,
    as long as they do NOT contradict the Reference Answer.

    D. EXTRA DETAILS / LENGTH (VERY SOFT)
    - Do NOT mark false just because the answer is long, includes irrelevant explanation, or adds general commentary.
    - Extra details must NOT be used to substitute missing required key points (Rule A still applies).

    E. INSUFFICIENT / REFUSAL
    - If the evaluated answer refuses or says "insufficient information" while the reference provides an answer -> false.

    OUTPUT FORMAT:
    Output ONLY one JSON object and NOTHING else:
    {{"judgement_result": true}}
    or
    {{"judgement_result": false}}
    """

    JUDGE_BOOL_SCHEMA = {
    "name": "judge_bool_schema",
    "schema": {
        "type": "object",
        "properties": {
        "judgement_result": {
            "type": "boolean",
            "description": "true if correct, false otherwise."
        }
        },
        "required": ["judgement_result"],
        "additionalProperties": False
    }
    }

    system_prompt = "You are a fair and unbiased evaluator."
    response1 = get_llm_response(system_prompt, prompt,JUDGE_BOOL_SCHEMA,model="gpt-5-2025-08-07", json_format=True)
    bool_result1 = parse_bool_from_json(response1)
    logger.info(f"FINAL_ANSWER judgement_result: {bool_result1}")
    if bool_result1:
        corr1 = 1
    else:
        corr1 = 0

    response2 = get_llm_response(system_prompt, prompt,JUDGE_BOOL_SCHEMA,model="gemini-3-pro-preview", json_format=True)
    bool_result2 = parse_bool_from_json(response2)
    logger.info(f"FINAL_ANSWER judgement_result: {bool_result2}")
    if bool_result2:
        corr2 = 1
    else:
        corr2 = 0
    response3 = get_llm_response(system_prompt, prompt,JUDGE_BOOL_SCHEMA,model="gpt-4o-2024-11-20", json_format=True)
    bool_result3 = parse_bool_from_json(response3)
    logger.info(f"FINAL_ANSWER judgement_result: {bool_result3}")
    if bool_result3:
        corr3 = 1
    else:
        corr3 = 0
        
    # Calculate F1 for evidence clips
    # retrieved_clips = set(sample_idx)
    # golden_clips = set(golden_evidence_clip)
    # tp = len(retrieved_clips & golden_clips)
    # fp = len(retrieved_clips - golden_clips)
    # fn = len(golden_clips - retrieved_clips)
    # precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    # recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # corr1 = 0

    # Calculate F1 for evidence clips
    # retrieved_clips = set(sample_idx)
    # golden_clips = set(golden_evidence_clip)
    # tp = len(retrieved_clips & golden_clips)
    # fp = len(retrieved_clips - golden_clips)
    # fn = len(golden_clips - retrieved_clips)
    # precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    # recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # F1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
    cost = cost
    return corr1,corr2,corr3,cost

if __name__ == "__main__":
    pass