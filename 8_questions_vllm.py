# 8_questions_vllm.py


import pyarrow as pa
import pyarrow.parquet as pq
import time
import re
import os
from pathlib import Path
from vllm import LLM, SamplingParams


# Starting row number for resuming processing
START = 100_352
START_FILE = START // 100_352
PROCESS_FILE_SIZE = 100_000

# Input and output paths for samples and generated questions
BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "samples.parquet"
OUTPUT_DIR = BASE_DIR / "question_parquets"

# Number of rows to buffer before flushing to parquet
WRITE_BATCH_SIZE = 2048

# Model generation batch settings
BATCH_SIZE = 256
MAX_NEW_TOKENS = 32

MODEL_NAME = "mistralai/Mistral-7B-v0.1"

# Filters to remove XML-like tag output and disallowed token ids
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

# Prompt suffix that closes the context and starts the question tag
SUFFIX = "</context>\n<question>\n"

BANNED_ANYWHERE = re.compile(r'\b(question|answer|context)\b')


def extract_last_segment(q):
    # Get the last non-empty line from model output
    if not isinstance(q, str):
        return None
    parts = [p.strip() for p in q.split("\n") if p.strip()]
    return parts[-1] if parts else None


def is_valid(q: str) -> bool:
    # Validate that the generated question is well-formed
    if not q:
        return False

    if not q.endswith("?"):
        return False

    if len(q) < 10:
        return False

    if q.count('?') > 1:
        return False
    
    if q.count('\n'):
        return False

    lower = q.lower()

    if BANNED_ANYWHERE.search(lower):
        return False

    return True


def validate(input_path):
    # Clean and validate the generated question field in a parquet file
    table = pq.read_table(input_path)

    cleaned_questions = [extract_last_segment(q) for q in table.column("question").to_pylist()]
    table = table.set_column(
        table.schema.get_field_index("question"),
        "question",
        pa.array(cleaned_questions)
    )

    valid_mask = [is_valid(q) for q in cleaned_questions]
    table = table.set_column(
        table.schema.get_field_index("is_valid"),
        "is_valid",
        pa.array(valid_mask)
    )

    tmp_path = str(input_path) + ".tmp"
    writer = pq.ParquetWriter(tmp_path, table.schema, compression="zstd")

    writer.write_table(table)
    writer.close()

    os.replace(tmp_path, input_path)


def main():
    # Ensure output directory exists before generating questions
    OUTPUT_DIR.mkdir(exist_ok=True)

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

    parquet_file = pq.ParquetFile(INPUT_PATH)
    total_rows = parquet_file.metadata.num_rows

    writer = None
    processed_total = 0

    flush_counter = 0
    file_counter = START_FILE
    processed_in_file = 0

    batched_outputs = []

    start = time.perf_counter()

    for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE):
        # if (processed_total < START):
        #     continue

        processed_total += batch.num_rows
        processed_in_file += batch.num_rows
        flush_counter += batch.num_rows

        ids = batch.column("id")
        texts = batch.column("text").to_pylist()

        prompts = [PREFIX + text + SUFFIX for text in texts]

        # Generate question text for each sampled context
        outputs = llm.generate(prompts, sampling_params)
        decoded = [o.outputs[0].text.strip() if o.outputs else "" for o in outputs]

        out_table = pa.table({
            "id": ids,
            "is_valid": pa.array([False] * len(decoded)),
            "text": texts,
            "question": decoded,
        })

        batched_outputs.append(out_table)

        if flush_counter >= WRITE_BATCH_SIZE or processed_total >= total_rows:
            flush_counter = 0

            combined = pa.concat_tables(batched_outputs)

            if writer is None:
                out_path = os.path.join(OUTPUT_DIR, f"question_batch_{file_counter}.parquet")
                writer = pq.ParquetWriter(
                    out_path,
                    combined.schema,
                    compression="zstd"
                )
            
            # Write the current buffered batch to parquet
            writer.write_table(combined)
            batched_outputs.clear()

            if processed_in_file >= PROCESS_FILE_SIZE:
                writer.close()
                writer = None

                processed_in_file = 0
                file_counter += 1

                print(f"Switched to file {file_counter}")

            pct = (processed_total / total_rows) * 100
            print(f"{processed_total}/{total_rows} ({pct:.2f}%) - Time: {time.perf_counter() - start:.2f} seconds")
            start = time.perf_counter()

    # Validate all generated question files after generation completes
    for fname in os.listdir(OUTPUT_DIR):
        validate(os.path.join(OUTPUT_DIR, fname))


if __name__ == "__main__":
    main()