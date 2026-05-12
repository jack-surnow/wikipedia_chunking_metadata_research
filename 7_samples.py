# 7_samples.py


import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

INPUT_DIR = "chunk_parquets"
OUTPUT_PATH = "samples.parquet"

# Number of rows present in each chunk parquet file
ROWS_PER_FILE = 100_000

# Total number of chunk rows across all input files
TOTAL_ROWS = 32_627_304

# Number of random samples to select
NUM_QUESTIONS = 1_000_000
SEED = 42

np.random.seed(SEED)

# Sample a fixed set of global row ids without replacement
sampled_ids = np.random.choice(TOTAL_ROWS, size=NUM_QUESTIONS, replace=False)
sampled_ids.sort()

file_to_indices = {}

# Map each sampled row to its source parquet file and local index
for qid in sampled_ids:
    file_idx = qid // ROWS_PER_FILE
    local_idx = qid % ROWS_PER_FILE

    file_to_indices.setdefault(file_idx, []).append((qid, local_idx))


writer = None

# Read each source parquet file and extract the sampled rows
for file_idx, items in file_to_indices.items():

    file_path = os.path.join(INPUT_DIR, f"chunk_batch_{file_idx}.parquet")

    table = pq.read_table(file_path)

    items.sort(key=lambda x: x[1])
    local_indices = [x[1] for x in items]
    global_ids = [x[0] for x in items]

    # Take only the sampled rows and keep text plus global id
    subset = table.take(pa.array(local_indices))

    subset = subset.select(["text"])

    subset = subset.append_column("id", pa.array(global_ids))

    subset = subset.select(["id", "text"])

    if writer is None:
        # Create the output parquet writer on first file
        writer = pq.ParquetWriter(
            OUTPUT_PATH,
            subset.schema,
            compression="zstd"
        )

    writer.write_table(subset)

if writer:
    writer.close()
