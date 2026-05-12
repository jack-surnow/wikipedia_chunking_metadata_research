# 8_questions.py


import pyarrow.parquet as pq
import pyarrow as pa
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM


# Input sample file to generate questions from
INPUT_PATH = "samples.parquet"

# Output parquet file for generated questions
OUTPUT_PATH = "questions.parquet"

# Number of examples to process per model batch
BATCH_SIZE = 16

# Maximum generated tokens for each question
MAX_NEW_TOKENS = 64

# Model identifier for generation
MODEL_NAME = "mistralai/Mistral-7B-v0.1"


# Prompt template for generating a search-style question from passage text
PROMPT = """You are generating questions for a search engine query dataset.

Read the passage below and write a single question that someone might type into Google to find this information. The question must make sense on its own with no context.

Rules:
- Do NOT use words like "passage", "text", "article", "excerpt", "author", "essay", or "according to"
- Do NOT reference that any source material exists
- Do NOT ask questions that could be answered by ctrl+f searching the text unless there is no option

Passage:
{passage}

Question:"""


# Require GPU for faster generation
assert torch.cuda.is_available(), "CUDA not available"

# Load tokenizer and configure padding for left-side generation
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

# Load the causal language model in mixed precision and move it to GPU
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

assert next(model.parameters()).is_cuda, "Model is not on GPU"


# Open input parquet file and count total rows for progress tracking
parquet_file = pq.ParquetFile(INPUT_PATH)
total_rows = parquet_file.metadata.num_rows

writer = None
processed = 0

start = end = time.perf_counter()

with torch.no_grad():
    for i, batch in enumerate(parquet_file.iter_batches(batch_size=BATCH_SIZE)):
        start = end

        batch_size_actual = batch.num_rows
        processed += batch_size_actual

        ids = batch.column("id").to_pylist()
        texts = batch.column("text").to_pylist()

        prompts = [PROMPT.format(passage=text) for text in texts]

        # Tokenize the prompt batch for model input
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )

        inputs = {k: v for k, v in inputs.items()}

        # Generate questions from the model with sampling behavior
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=False
        )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        out_table = pa.table({
            "id": ids,
            "question": decoded
        })

        if writer is None:
            # Open output writer on first batch
            writer = pq.ParquetWriter(
                OUTPUT_PATH,
                out_table.schema,
                compression="zstd"
            )

        writer.write_table(out_table)


        end = time.perf_counter()
        pct = (processed / total_rows) * 100
        print(f"{processed}/{total_rows} ({pct:.2f}%) - Time: {end - start:.2f} seconds")

if writer:
    writer.close()