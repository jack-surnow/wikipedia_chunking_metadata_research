# Wikipedia Metadata Ablation for Dense Retrieval

**Which Wikipedia metadata should be prepended to a passage before embedding it?** This project compares 15 metadata configurations using a shared chunk corpus, a BGE embedding model, FAISS retrieval, and one million synthetic questions, evaluated both directly and after LLM rewording.

The main finding is that **compact title metadata improves retrieval, but more metadata is not consistently better**. Page + parent + section titles produce the highest Recall@10 on direct questions; page title alone leads on reworded questions. Section title alone gives the highest MRR in both settings.

The repository contains the pipeline scripts and reported result tables. It does **not** include the Wikipedia corpus, generated questions, embeddings, indexes, or raw evaluation outputs. The results below are reported experiments, not benchmarks rerun from a packaged environment.

## Findings at a glance

| Query set | No-metadata Recall@10 | Highest Recall@10 | Configuration | Absolute improvement |
| --- | ---: | ---: | --- | ---: |
| Direct synthetic questions | 28.823% | 30.815% | Page + parent + section titles (test 10) | +1.992 percentage points |
| Reworded synthetic questions | 26.715% | 28.841% | Page title (test 3) | +2.126 percentage points |

- **Page title is a strong, simple choice.** On direct questions, it reaches 30.752% Recall@10, only 0.063 percentage points below test 10. The tables do not establish statistical significance for this small difference.
- **The best configuration depends on the metric.** Section title alone leads MRR@1000: 0.21719 for direct questions and 0.19917 for reworded questions.
- **Additional fields do not guarantee better retrieval.** Adding depth and position to the three-title configuration lowers Recall@10 in both query sets.
- **Rewording makes source-chunk retrieval harder in these runs.** It introduces a second query condition, but does not establish performance on real user questions.

## Experimental setup

This is a **metadata ablation with a fixed chunking pipeline**, not a comparison of different chunking algorithms or an end-to-end answer-quality benchmark.

| Component | Configuration in the scripts |
| --- | --- |
| Source | English Wikipedia; the extraction script points to the `enwiki-20260301-pages-articles-multistream.xml.bz2` dump |
| Corpus size | Sampling is configured for 32,627,304 chunks; generated data is not included |
| Chunking | Paragraph/newline/sentence-aware splitting; target range 800–1600 characters and 100-character overlap |
| Metadata | Prepended before embedding, capped at 200 characters, separated from passage text by a blank line |
| Passage and query encoder | `BAAI/bge-small-en-v1.5`; 384-dimensional, normalized embeddings; maximum sequence length 512 tokens |
| Vector index | FAISS IVF-PQ with inner product on normalized vectors; 32,768 lists, 48 PQ subquantizers, 8 bits per code |
| Index training | Up to 1,500,000 sampled vectors; file order is shuffled without a fixed seed |
| Retrieval | `nprobe=32`; top 1,000 candidates per query |
| Query sample | 1,000,000 source chunks sampled without replacement; NumPy seed 42 |
| Query generation / rewording | `mistralai/Mistral-7B-v0.1`, with formatting checks and regeneration |
| Relevance label | The original source chunk's ID; other potentially relevant chunks are not labeled |

### What the metrics mean

- **Recall@K:** fraction of queries whose source chunk appears in the first K results. With one labeled chunk per query, this is also a hit rate.
- **MRR@1000:** mean reciprocal rank of the source chunk within the top 1,000 results; a missing source chunk contributes zero. The evaluator writes this value under the JSON key `mrr`.
- Full result tables use fractions: `0.30815` means **30.815%**. Differences between percentages are reported as **percentage points**.

### Metadata configurations

Test numbers below follow `NEW_FIELDS` in [`4_metadata.py`](4_metadata.py), which is the mapping used by `gen_test()`. The alternate `FULL_FIELDS` mapping is not used by that function and assigns different meanings to some test numbers.

| Test | Description |
|------|-------------|
| 1 | No metadata (baseline 1) |
| 2 | Full hierarchical path (baseline 2) |
| 3 | Page title |
| 4 | Section title |
| 5 | Parent section title |
| 6 | Section depth level |
| 7 | Section position within page |
| 8 | Page title + section title |
| 9 | Parent title + section title |
| 10 | Page title + parent title + section title |
| 11 | Section depth level + position |
| 12 | Page title + section title + depth level |
| 13 | Page title + section title + position |
| 14 | Page title + section title + depth level + position |
| 15 | Page title + parent title + section title + depth level + position |

`position` is a categorical field (`start`, `mid`, or `end`), not a raw section index. Missing title values are omitted. Metadata uses labeled lines such as `page:`, `parent:`, and `section:`.

For example, test 10 creates an embedding input of this form:

```text
page: Example article
parent: Example parent section
section: Example subsection

Original passage text...
```

## Full results

The same source-question set is used across metadata configurations within each query condition. Direct questions are generated from unaugmented source passages; the reworded condition paraphrases those questions. Neither condition uses human relevance judgments.

### Direct synthetic questions

|   Test |   Recall@10 |   Recall@100 |   Recall@1000 |     MRR@1000 |
|-------:|------------:|-------------:|--------------:|--------:|
|      1 |     0.28823 |      0.37744 |       0.44729 | 0.20552 |
|      2 |     0.28636 |      0.37464 |       0.44256 | 0.20434 |
|      3 |     0.30752 |      0.3999  |       0.46907 | 0.21467 |
|      4 |     0.30134 |      0.39051 |       0.45849 | **0.21719** |
|      5 |     0.28911 |      0.37783 |       0.44644 | 0.20539 |
|      6 |     0.29903 |      0.38271 |       0.44744 | 0.21494 |
|      7 |     0.29866 |      0.38677 |       0.45282 | 0.21196 |
|      8 |     0.30423 |      0.39518 |       0.46464 | 0.21355 |
|      9 |     0.29653 |      0.386   |       0.45428 | 0.2125  |
|     10 |     **0.30815** |      **0.40006** |       **0.4692**  | 0.21582 |
|     11 |     0.29164 |      0.37642 |       0.44017 | 0.20812 |
|     12 |     0.30531 |      0.39581 |       0.46299 | 0.21504 |
|     13 |     0.30223 |      0.39513 |       0.46393 | 0.21077 |
|     14 |     0.30112 |      0.39273 |       0.4612  | 0.21243 |
|     15 |     0.3028  |      0.39383 |       0.46259 | 0.21092 |

### Reworded synthetic questions

|   Test |   Recall@10 |   Recall@100 |   Recall@1000 |     MRR@1000 |
|-------:|------------:|-------------:|--------------:|--------:|
|      1 |     0.26715 |      0.35626 |       0.42712 | 0.18797 |
|      2 |     0.26675 |      0.35452 |       0.42389 | 0.18798 |
|      3 |     **0.28841** |      **0.37993** |       0.44994 | 0.19865 |
|      4 |     0.28069 |      0.37045 |       0.43982 | **0.19917** |
|      5 |     0.26949 |      0.35762 |       0.42736 | 0.18845 |
|      6 |     0.2785  |      0.36251 |       0.42875 | 0.1975  |
|      7 |     0.27807 |      0.36619 |       0.4335  | 0.19432 |
|      8 |     0.28324 |      0.37476 |       0.44568 | 0.19653 |
|      9 |     0.27609 |      0.36467 |       0.43452 | 0.19524 |
|     10 |     0.28749 |      0.37901 |       **0.44999** | 0.19873 |
|     11 |     0.27174 |      0.35782 |       0.42292 | 0.19181 |
|     12 |     0.28514 |      0.37486 |       0.4437  | 0.19828 |
|     13 |     0.28206 |      0.3748  |       0.44572 | 0.19438 |
|     14 |     0.28135 |      0.37134 |       0.44145 | 0.19582 |
|     15 |     0.28267 |      0.37427 |       0.4438  | 0.19475 |

## Pipeline and files

Run scripts from the repository root: several paths are relative to the current working directory. Data preparation produces one shared corpus; metadata augmentation, passage embedding, and indexing run separately for each selected configuration. Questions are generated once and reused across configurations.

| Stage | Script | Output / purpose |
| --- | --- | --- |
| Extract | [`1_raw.py`](1_raw.py) | `raw_batches/`: main-namespace article text, filtering redirects and disambiguation titles |
| Structure | [`2_sections.py`](2_sections.py) | `section_batches/`: cleaned sections and hierarchy metadata |
| Chunk | [`3_chunks.py`](3_chunks.py) | `chunk_parquets/`: passage text, metadata, and global chunk IDs |
| Augment | [`4_metadata.py`](4_metadata.py) | `chunk_augmented_<test>/`: metadata-prefixed passage text |
| Embed passages | [`5_embeddings.py`](5_embeddings.py) or [`5_embeddings_vllm.py`](5_embeddings_vllm.py) | `embeddings_<test>/`: chunk IDs and embeddings |
| Index | [`6_faiss.py`](6_faiss.py) | `faiss_indexes/ivfpq_index_<test>.faiss` |
| Sample | [`7_samples.py`](7_samples.py) | `samples.parquet`: shared source-chunk sample |
| Generate questions | [`8_questions.py`](8_questions.py) or [`8_questions_vllm.py`](8_questions_vllm.py) | `question_parquets/`: questions linked to source IDs |
| Refine | [`9_refine_vllm.py`](9_refine_vllm.py) | Regenerates questions that fail formatting checks, in place |
| Reword | [`10_reword_vllm.py`](10_reword_vllm.py) | Adds reworded questions and validation flags to the same files |
| Embed queries | [`11_embed_questions.py`](11_embed_questions.py) | `question_embeddings/` or `reworded_embeddings/` |
| Evaluate | [`12_test.py`](12_test.py) | Per-configuration JSON metrics |

The vLLM passage embedder launches one worker per configured GPU. The SentenceTransformers passage and query embedders instead take a GPU ID as a positional argument and process that GPU's shard. Run every configured shard to produce complete outputs.

## Running the experiments

These are research scripts with configuration constants, rather than a single-command application. The checked-in defaults represent selected runs and **do not run all 15 configurations automatically**. A full Wikipedia-scale run requires GPU inference, substantial storage, and enough host memory for index training; hardware specifications and a validated dependency lockfile are not included.

### Dependencies

The scripts import `torch`, `transformers`, `sentence-transformers`, `vllm`, `faiss`, `numpy`, `pandas`, `pyarrow`, `lxml`, `mwparserfromhell`, and `nltk`. Install compatible versions for your CUDA/PyTorch environment. NLTK sentence-tokenization data is also required. Exact package versions for the reported runs are not recorded here; in particular, the vLLM scripts use version-sensitive APIs.

### Configure before running

1. In `1_raw.py`, replace the local Windows `INPUT_PATH` with your downloaded Wikipedia dump. Use a consistent dump for all configurations.
2. Use the corrected path implementation for new test 2 runs; see the historical-results caveat above. Set `TESTS` consistently in `4_metadata.py`, your chosen passage embedder, `6_faiss.py`, and `12_test.py`. Use `list(range(1, 16))` for all 15 configurations.
3. Set GPU counts in the embedders and `tensor_parallel_size` in the vLLM generation/refinement/rewording scripts to match your hardware. Do not run both passage embedding implementations into the same output folders.
4. After chunking, update `TOTAL_ROWS` and `ROWS_PER_FILE` in `7_samples.py` to match the generated corpus. The sampler assumes contiguous IDs and sequential `chunk_batch_<n>.parquet` files. Preserve those artifacts across configurations: the chunker assigns IDs using unsorted directory traversal, so regenerating chunks can change ID assignments.
5. For a fresh vLLM question-generation run, set `START=0` and `START_FILE=0` in `8_questions_vllm.py`. Its current resume offset does not skip input rows because that logic is commented out. Use fresh output folders rather than relying on resume behavior.
6. In `10_reword_vllm.py`, `NEW=True` resets reworded output fields in the existing question files. Back up those files before restarting a completed run.
7. Set `REWORDED_QUESTIONS` consistently in `11_embed_questions.py` and `12_test.py`. Choose distinct evaluator `OUTPUT_DIR` values for direct and reworded runs: changing the Boolean alone does not change the currently hard-coded output directory.
8. Check that all sampled IDs have nonempty, valid questions before embedding. The evaluator does not filter question-validity flags, so unfinished refinement can affect the results.

### Execution order

After the configuration checks above, the vLLM route is:

```bash
python 1_raw.py
python 2_sections.py
python 3_chunks.py
python 4_metadata.py
python 5_embeddings_vllm.py
mkdir -p faiss_indexes
python 6_faiss.py
python 7_samples.py
python 8_questions_vllm.py
python 9_refine_vllm.py
python 10_reword_vllm.py
```

For query embeddings with `GPU_COUNT=2`, run both shards (sequentially or in separate terminals), then evaluate:

```bash
python 11_embed_questions.py 0
python 11_embed_questions.py 1
python 12_test.py
```

Repeat query embedding and evaluation with `REWORDED_QUESTIONS=False` and `True`, using separate metric output directories. For the SentenceTransformers passage-embedding alternative, use `python 5_embeddings.py 0` and `python 5_embeddings.py 1` instead of `5_embeddings_vllm.py`, after aligning `TESTS` and `GPU_COUNT`.

Start with a small corpus to validate paths, ID alignment, and output schemas. Reduce sampling counts and FAISS training/index parameters to suit that corpus; a smoke run with altered settings does not reproduce the -scale results above.

## Interpretation and limitations

- **Synthetic-query evaluation:** questions are derived from the same passages being retrieved. Paraphrasing changes wording but does not remove this dependence or validate real-user performance. Formatting checks do not guarantee answerability, semantic equivalence, or absence of leakage.
- **One labeled chunk per query:** another retrieved passage may answer the question correctly and still count as a miss. These results measure source-chunk recovery, not answer correctness.
- **Approximate retrieval:** IVF-PQ compression and search settings contribute to the measured scores. They do not isolate embedding quality from ANN search error.
- **Limited generalization:** the reported ablation uses one Wikipedia corpus, one embedding model, and one chunking setup. Different domains, encoders, or token budgets may favor different metadata.
- **Uncertainty is not quantified:** no confidence intervals or repeated-run variance are reported. Small differences should not be treated as established superiority.
- **Reproduction artifacts are incomplete:** generated datasets, raw metric JSON files, environment versions, and hardware/run manifests are not checked in. The current scripts and README alone do not establish exact reproduction of the historical tables.

To make future runs independently reproducible, preserve the dump identifier/checksum, code revision, dependency versions, hardware, all configuration constants and random seeds, shared chunk/question artifacts, validity counts, and per-test metric JSON files.

## Regression tests

With `pandas` and `pyarrow` installed, run:

```bash
python -m unittest discover -s tests -v
```

These tests cover path inclusion, hierarchy order, length boundaries, truncation, unchanged non-path inputs, and a small test 2 Parquet round trip. They do not rerun the retrieval experiments.
