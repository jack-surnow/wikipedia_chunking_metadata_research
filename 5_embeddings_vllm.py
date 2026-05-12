# 5_embedding_vllm.py


import os
import glob
from pathlib import Path

import multiprocessing as mp
import traceback

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds



GPU_COUNT = 2

# Tests to process for embedding generation
TESTS = [4, 5, 6, 7, 9, 11]

# Not currently used by embedder, but kept for future batch control
BATCH_SIZE = 32
MAX_LENGTH = 512

# Embedding model name
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Base directory for input and output paths
BASE_DIR = Path(__file__).resolve().parent


def generate(gpu_id, llm, output_dir, file_list):
    # Generate embeddings for each parquet file assigned to this GPU
    for file in file_list:
        filename = os.path.basename(file)
        out_path = os.path.join(output_dir, filename)

        print(f"[GPU {gpu_id}] {filename}", flush=True)

        dataset = ds.dataset(file, format="parquet")
        table = dataset.to_table(columns=["id", "text"])
        
        ids = table.column("id").to_numpy()
        texts = table.column("text").to_pylist()

        # Embed texts with tokenizer truncation to max length
        outputs = llm.embed(
            texts,
            tokenization_kwargs={
                "truncation": True,
                "max_length": 512
            }
        )
        # Convert output embeddings to numpy and normalize to unit length
        embeddings = np.asarray(
            [o.outputs.embedding for o in outputs],
            dtype=np.float32
        )

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        )

        embeddings = embeddings / norms

        # Build parquet arrays for ids and fixed-size embeddings
        id_array = pa.array(ids, type=pa.int64())
        emb_array = pa.FixedSizeListArray.from_arrays(
            pa.array(embeddings.ravel(), type=pa.float32()),
            embeddings.shape[1]
        )

        table_out = pa.Table.from_arrays(
            [id_array, emb_array],
            names=["id", "embedding"]
        )

        writer = pq.ParquetWriter(
            out_path,
            table_out.schema,
            compression="zstd"
        )

        writer.write_table(table_out)
        writer.close()


def worker(gpu_id):
    # Worker process bound to one GPU that loads model and processes assigned files
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        from vllm import LLM

        llm = LLM(
            model=MODEL_NAME,
            runner="pooling",
            dtype="float16",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            enforce_eager=False,
            max_model_len=512,
        )

        for test in TESTS:
            print(f"Test {test}", flush=True)

            input_dir = BASE_DIR / f"chunk_augmented_{test}"
            output_dir = BASE_DIR / f"embeddings_{test}"

            os.makedirs(output_dir, exist_ok=True)

            files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
            num_files = len(files)

            # Divide files evenly across GPUs, distributing remainder files one per GPU
            files_per_gpu = num_files // GPU_COUNT
            remainder = num_files % GPU_COUNT

            start = gpu_id * files_per_gpu + min(gpu_id, remainder)
            end = start + files_per_gpu + (1 if gpu_id < remainder else 0)

            generate(gpu_id, llm, output_dir, files[start:end])

            print("\n\n\n", flush=True)
        
    except Exception as e:
        traceback.print_exc()
        raise


def main():
    # Launch separate worker processes for each GPU
    mp.set_start_method("spawn", force=True)

    processes = []

    for gpu_id in range(GPU_COUNT):
        p = mp.Process(target=worker, args=(gpu_id,))

        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()