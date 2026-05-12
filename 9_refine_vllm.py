# 9_refine_vllm.py


import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import re
import os
from pathlib import Path
from vllm import LLM, SamplingParams


# Number of examples to send to the model at once
BATCH_SIZE = 256

# Maximum tokens to generate for each refined question
MAX_NEW_TOKENS = 32

# Model used for question refinement
MODEL_NAME = "mistralai/Mistral-7B-v0.1"

# Filters to discourage XML-like output from the model
BAD_WORDS = ["<", "</", " <", " </"]
BAD_TOKEN_IDS = [[28789], [700], [523], [1867]]

PREFIX = """Convert the context into a Google search query question.

The question must:
- be natural
- reflect the main topic
- be a single sentence
- end with a question mark

Output the question inside <question> tags before providing the answer inside <answer> tags.

<context>
""" 

# Prompt suffix that ends the context and begins the question field
SUFFIX = "</context>\n<question>\n"

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "question_parquets"

# Number of generated rows to buffer before writing parquet
WRITE_BATCH_SIZE = 2048

BANNED_ANYWHERE = re.compile(r'\b(question|answer|context)\b')


def extract_last_segment(q):
    # Extract the final non-empty line from model output
    if not isinstance(q, str):
        return None
    parts = [p.strip() for p in q.split("\n") if p.strip()]
    return parts[-1] if parts else None


def is_valid(q: str) -> bool:
    # Validate that the refined question is a plausible standalone query
    if not q:
        return False

    if not q.endswith("?"):
        return False

    if len(q) < 10:
        return False

    if q.count('?') > 1:
        return False
    
    if '\n' in q:
        return False

    lower = q.lower()

    if BANNED_ANYWHERE.search(lower):
        return False

    return True
        

def regenerate(iteration, llm, sampling_params, input_path) -> bool:
    # Regenerate invalid questions in-place and return True once all are valid
    tmp_path = str(input_path) + ".tmp"

    table = pq.read_table(input_path)

    invalid_mask = pc.equal(table["is_valid"], False)
    invalid_table = table.filter(invalid_mask)

    initial_count = invalid_mask.to_pylist().count(True)
    print(f"iteration {iteration} - initial: {initial_count}")

    ids = invalid_table.column("id").to_pylist()
    texts = invalid_table.column("text").to_pylist()

    # Construct prompts only for invalid questions
    prompts = [PREFIX + text + SUFFIX for text in texts]
    decoded = []

    for i in range(0, len(prompts), BATCH_SIZE):
        batch_prompts = prompts[i:i + BATCH_SIZE]

        outputs = llm.generate(batch_prompts, sampling_params)

        batch_decoded = [
            o.outputs[0].text.strip() if o.outputs else ""
            for o in outputs
        ]

        decoded.extend(batch_decoded)

    new_questions = [extract_last_segment(q) for q in decoded]
    id_to_question = dict(zip(ids, new_questions))

    full_ids = table.column("id").to_pylist()
    old_questions = table.column("question").to_pylist()

    updated_questions = [id_to_question.get(i, old_q) for i, old_q in zip(full_ids, old_questions)]
    updated_valid = [is_valid(q) for q in updated_questions]

    updated_count = updated_valid.count(False)
    print(f"iteration {iteration} - updated: {updated_count}")

    if updated_count == 0:
        return True
    
    invalid_entries = [(i, q) for i, q, valid in zip(full_ids, updated_questions, updated_valid) if not valid]
    for bad_id, bad_question in invalid_entries:
        print(f"id={bad_id} | question={repr(bad_question)}")

    # Replace the parquet columns with updated questions and validation flags
    table = table.set_column(table.schema.get_field_index("question"), "question", pa.array(updated_questions))
    table = table.set_column(table.schema.get_field_index("is_valid"), "is_valid", pa.array(updated_valid))

    pq.write_table(table, tmp_path, compression="zstd")
    os.replace(tmp_path, input_path)

    return False


def main():
    # Initialize the vLLM model for refinement passes
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=2,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        enforce_eager=False,
        max_num_batched_tokens=8192
    )

    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.8,
        top_p=0.95,
        stop=["?"],
        include_stop_str_in_output=True,
        _bad_words_token_ids=BAD_TOKEN_IDS,
    )

    # Iterate over generated question files and refine invalid outputs
    for fname in os.listdir(INPUT_DIR):
        for i in range(0, 50):
            if (regenerate(i, llm, sampling_params, os.path.join(INPUT_DIR, fname))):
                break


if __name__ == "__main__":
    main()
