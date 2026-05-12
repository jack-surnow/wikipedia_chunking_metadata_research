# faiss.py


import os
import glob
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import faiss
import random
import time

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "faiss_indexes"

TESTS = [15]

# FAISS index parameters: embedding dimension, number of partitions, PQ codes, bits per code
DIM = 384
NLIST = 32768
M = 48
NBITS = 8

TRAIN_SAMPLES = 1_500_000
BATCH_SIZE = 100_000


def sample_training_data(files, target_samples):
    # Randomly sample embeddings from parquet files for index training
    collected = []
    total = 0

    random.shuffle(files)

    for file in files:
        dataset = ds.dataset(file, format="parquet")
        table = dataset.to_table(columns=["embedding"])

        embeddings = np.vstack(table.column("embedding").to_numpy())
        embeddings = embeddings.astype(np.float32)

        collected.append(embeddings)
        total += embeddings.shape[0]

        print(f"[TRAIN] collected {total}")

        if total >= target_samples:
            break

    train_data = np.vstack(collected)[:target_samples]

    # Normalize for inner product metric
    faiss.normalize_L2(train_data)
    return train_data


def create_index(test_num):
    # Build and save FAISS IVF-PQ index for the given test number
    t1 = time.perf_counter()

    input_dir = f"embeddings_{test_num}"
    index_path = OUTPUT_DIR / f"ivfpq_index_{test_num}.faiss"

    # Create IVF-PQ index with flat inner product quantizer
    quantizer = faiss.IndexFlatIP(DIM)

    index = faiss.IndexIVFPQ(
        quantizer,
        DIM,
        NLIST,
        M,
        NBITS,
        faiss.METRIC_INNER_PRODUCT
    )

    # Wrap to preserve original IDs
    index = faiss.IndexIDMap2(index)

    # Train the index with sampled data
    files = glob.glob(os.path.join(input_dir, "*.parquet"))

    print("Sampling training data...")
    train_data = sample_training_data(files, TRAIN_SAMPLES)

    print("Training index...")
    index.train(train_data)

    del train_data

    t2 = time.perf_counter()
    print(f"Training completed in {t2 - t1:.6f} seconds")

    # Add all vectors to the index
    print("Adding vectors...")

    for file in files:
        dataset = ds.dataset(file, format="parquet")
        table = dataset.to_table(columns=["id", "embedding"])

        ids = table.column("id").to_numpy().astype(np.int64)

        embeddings = np.vstack(table.column("embedding").to_numpy())
        embeddings = embeddings.astype(np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add in batches to prevent memory spikes
        for i in range(0, len(ids), BATCH_SIZE):
            batch_ids = ids[i:i+BATCH_SIZE]
            batch_emb = embeddings[i:i+BATCH_SIZE]

            index.add_with_ids(batch_emb, batch_ids)

        print(f"Added {len(ids)} from {file} | total={index.ntotal}")

    t3 = time.perf_counter()
    print(f"Adding completed in {t3 - t2:.6f} seconds")

    # Save index to disk
    print("Saving index...")
    faiss.write_index(index, index_path)

    t4 = time.perf_counter()
    print(f"Index saved in {t4 - t3:.6f} seconds")


def main():
    for num in TESTS:
        create_index(num)


if __name__ == "__main__":
    main()