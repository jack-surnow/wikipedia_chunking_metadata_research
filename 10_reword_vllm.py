# 10_reword_vllm.py


import os
import glob
from pathlib import Path

import re
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

from vllm import LLM, SamplingParams


# Whether to initialize or reset reworded outputs before generation
NEW = True

# Maximum number of regeneration passes per file
ITERATIONS = 50

# Model generation batch settings
BATCH_SIZE = 256
MAX_NEW_TOKENS = 32

# Model used for rewording questions
MODEL_NAME = "mistralai/Mistral-7B-v0.1"

# Filters to discourage XML-like tokens and bad sequences in output
BAD_WORDS = ["<", "</", " <", " </"]
BAD_TOKEN_IDS = [[28789], [700], [523], [1867]]

PREFIX = """Reword the following search query question.

The rewritten question must:
- preserve the exact meaning and intent
- NOT introduce new information
- be a natural Google search query
- be a single sentence
- end with a question mark
- avoid repeating original phrasing

Return ONLY the rewritten question inside <rewritten_question> tags.

<original_question>
"""

SUFFIX = """
</original_question>
<rewritten_question>"""

# Input and output directories are the same; files are updated in place
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "question_parquets"
OUTPUT_DIR = BASE_DIR / "question_parquets"

# Reject generated outputs that contain metadata or tag words
BANNED_ANYWHERE = re.compile(r'\b(question|answer|context|original_question|rewritten_question)\b')


# Initialize or reset reworded generation state for all parquet files
def initialize_output_dir(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    for input_path in files:

        fname = os.path.basename(input_path)
        output_path = os.path.join(output_dir, fname)

        table = pq.read_table(input_path)

        is_valid_reworded = pa.array([False] * table.num_rows)
        reworded = pa.array([""] * table.num_rows)

        # Creating new columns if they don't exist, otherwise resetting them
        if "is_valid_reworded" in table.schema.names:
            table = table.set_column(
                table.schema.get_field_index("is_valid_reworded"),
                "is_valid_reworded",
                is_valid_reworded
            )
        else:
            table = table.append_column("is_valid_reworded", is_valid_reworded)

        # Add or reset reworded column
        if "reworded" in table.schema.names:
            table = table.set_column(
                table.schema.get_field_index("reworded"),
                "reworded",
                reworded
            )
        else:
            table = table.append_column("reworded", reworded)

        pq.write_table(table, output_path, compression="zstd")


# Obtain last line of the generated output
def extract_last_segment(q):
    if not isinstance(q, str):
        return None
    parts = [p.strip() for p in q.split("\n") if p.strip()]
    return parts[-1] if parts else None


# Check if the generated output is valid
def is_valid(q: str) -> bool:
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
        

def regenerate(iteration, llm, sampling_params, output_path) -> bool:
    tmp_path = str(output_path) + ".tmp"

    table = pq.read_table(output_path)

    # Identify rows that still need rewording
    invalid_mask = pc.equal(table["is_valid_reworded"], False)
    invalid_table = table.filter(invalid_mask)

    initial_count = invalid_mask.to_pylist().count(True)
    print(f"iteration {iteration} - initial: {initial_count}")

    ids = invalid_table.column("id").to_pylist()
    questions = invalid_table.column("question").to_pylist()

    # Build prompts for each invalid question
    prompts = [PREFIX + question + SUFFIX for question in questions]
    decoded = []

    for i in range(0, len(prompts), BATCH_SIZE):
        batch_prompts = prompts[i:i + BATCH_SIZE]

        outputs = llm.generate(batch_prompts, sampling_params)

        batch_decoded = [
            o.outputs[0].text.strip() if o.outputs else ""
            for o in outputs
        ]

        decoded.extend(batch_decoded)

    new_reworded = [extract_last_segment(q) for q in decoded]
    id_to_question = dict(zip(ids, new_reworded))

    full_ids = table.column("id").to_pylist()
    old_reworded = table.column("reworded").to_pylist()

    updated_reworded = [id_to_question.get(i, old_rw) for i, old_rw in zip(full_ids, old_reworded)]
    updated_valid = [is_valid(q) for q in updated_reworded]

    updated_count = updated_valid.count(False)
    print(f"iteration {iteration} - updated: {updated_count}")
    
    if updated_count == 0:
        return True

    reworded_col = pa.array(updated_reworded)
    valid_col = pa.array(updated_valid)

    # Update table columns with new reworded text and validity flags
    table = table.set_column(table.schema.get_field_index("reworded"), "reworded", reworded_col)
    table = table.set_column(table.schema.get_field_index("is_valid_reworded"), "is_valid_reworded", valid_col)

    pq.write_table(table, tmp_path, compression="zstd")
    os.replace(tmp_path, output_path)

    return False


def main():
    # Reset generation state when starting a new run
    if NEW:
        initialize_output_dir(INPUT_DIR, OUTPUT_DIR)

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

    # Iterate over each output file and refine until all rewordings are valid
    for fname in os.listdir(OUTPUT_DIR):
        for i in range(ITERATIONS):
            if (regenerate(i, llm, sampling_params, os.path.join(OUTPUT_DIR, fname))):
                break


if __name__ == "__main__":
    main()
