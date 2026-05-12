import os
import glob
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

import torch
from sentence_transformers import SentenceTransformer


if not sys.argv[1]:
    raise Exception("must provide GPU id")

GPU_ID = int(sys.argv[1])
GPU_COUNT = 2

TESTS = [4]

BASE_DIR = Path(__file__).resolve().parent

BATCH_SIZE = 32
MAX_LENGTH = 512



def generate(gpu_id, model, output_dir, file_list):
    with torch.inference_mode():
        for file in file_list:
            # start = time.perf_counter()

            filename = os.path.basename(file)
            out_path = os.path.join(output_dir, filename)

            print(f"[GPU {gpu_id}] {filename}", flush=True)

            dataset = ds.dataset(file, format="parquet")

            table = dataset.to_table(columns=["id", "text"])
            ids = table.column("id").to_numpy()
            texts = table.column("text").to_pylist()
            

            embeddings = model.encode(
                texts,
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype(np.float32)

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
                compression="snappy"
            )

            writer.write_table(table_out)
            writer.close()

            # end = time.perf_counter()
            # print(f": {end - start:.6f} seconds")


def main():
    gpu_id = GPU_ID

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = SentenceTransformer(
        "BAAI/bge-small-en-v1.5",
        device=f"cuda:{gpu_id}",
        model_kwargs={"dtype": torch.float16}
    )
    model.max_seq_length = MAX_LENGTH

    for test in TESTS:
        print(f"Test {test}", flush=True)

        input_dir = BASE_DIR / f"chunk_augmented_{test}"
        output_dir = BASE_DIR / f"embeddings_{test}"

        os.makedirs(output_dir, exist_ok=True)

        files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
        num_files = len(files)

        files_per_gpu = num_files // GPU_COUNT
        remainder = num_files % GPU_COUNT

        start = gpu_id * files_per_gpu + min(gpu_id, remainder)
        end = start + files_per_gpu + (1 if gpu_id < remainder else 0)

        generate(gpu_id, model, output_dir, files[start:end])

        print("\n\n\n", flush=True)


if __name__ == "__main__":
    main()