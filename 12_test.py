# 12_test.py


import os
import glob
from pathlib import Path

import numpy as np
import json
import pyarrow.dataset as ds
import faiss


REWORDED_QUESTIONS = True

# Test indices to evaluate
TESTS = [1, 2, 3, 8, 10, 12 , 13, 14, 15]

BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "faiss_indexes"

# Use reworded question embeddings if available, otherwise use original
if REWORDED_QUESTIONS:
    QUESTION_EMBEDDING_DIR = BASE_DIR / "reworded_embeddings"
else:
    QUESTION_EMBEDDING_DIR = BASE_DIR / "question_embeddings"
OUTPUT_DIR = BASE_DIR / "retrieval_eval_reworded"

# Number of top results to retrieve for evaluation
TOP_K = 1000


def run_test(test):
    # Evaluate retrieval performance on test dataset using FAISS index
    
    # Load FAISS index and configure search parameters
    index_path = INDEX_DIR / f"ivfpq_index_{test}.faiss"
    index = faiss.read_index(str(index_path))

    ivf_index = faiss.extract_index_ivf(index)
    # nprobe controls how many cells to search in IVF
    ivf_index.nprobe = 32

    # Initialize metric counters
    total_questions = 0

    hits_at_10 = 0
    hits_at_100 = 0
    hits_at_1000 = 0

    reciprocal_rank_sum = 0.0

    # Process each batch of question embeddings
    files = sorted(glob.glob(os.path.join(QUESTION_EMBEDDING_DIR, "*.parquet")))
    for file in files:

        # Load question embeddings from parquet
        dataset = ds.dataset(file, format="parquet")
        table = dataset.to_table(columns=["id", "embedding"])

        ids = table.column("id").to_numpy()
        embeddings = np.vstack(table.column("embedding").to_numpy())
        embeddings = embeddings.astype(np.float32)

        # Search index for top-k neighbors of each question
        distances, retrieved_ids = index.search(embeddings, TOP_K)

        # Compute recall and MRR metrics
        for qid, ret_ids in zip(ids, retrieved_ids):
            total_questions += 1

            # Check if ground truth is in top-k results at different cutoffs
            if qid in ret_ids[:10]:
                hits_at_10 += 1

            if qid in ret_ids[:100]:
                hits_at_100 += 1

            if qid in ret_ids[:1000]:
                hits_at_1000 += 1

            # Calculate reciprocal rank for MRR metric
            matches = np.where(ret_ids == qid)[0]
            if len(matches) > 0:
                rank = int(matches[0]) + 1
                reciprocal_rank_sum += 1.0 / rank

        recall_at_10 = hits_at_10 / total_questions
        recall_at_100 = hits_at_100 / total_questions
        recall_at_1000 = hits_at_1000 / total_questions

        mrr = reciprocal_rank_sum / total_questions

        # Aggregate results into metrics dictionary
        metrics = {
            "test": test,
            "nprobe": ivf_index.nprobe,
            "top_k": TOP_K,

            "total_questions": total_questions,

            "recall_at_10": recall_at_10,
            "recall_at_100": recall_at_100,
            "recall_at_1000": recall_at_1000,

            "mrr": mrr,
        }

        # Save metrics to JSON file
        output_path = OUTPUT_DIR / f"test_{test}_metrics.json"

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(json.dumps(metrics, indent=4))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for test in TESTS:
        run_test(test)


if __name__ == "__main__":
    main()