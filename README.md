# Wikipedia Chunking & Metadata Research

This repository contains an end-to-end experimental pipeline for investigating how Wikipedia text chunking and metadata enrichment affect embedding quality, retrieval performance, and synthetic question generation for RAG-style systems.

The project builds a full pipeline from raw Wikipedia XML → structured chunks → metadata augmentation → embeddings → FAISS indexing → synthetic question generation → retrieval evaluation.

---

# Tested Metadata Configurations

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

Metadata fields were prepended to the beginning of each chunk prior to embedding. Each metadata field was placed on its own line, and the metadata block was separated from the chunk text by a blank line.

---

# Key Results

## 1. Direct Synthetic Queries (Ground-Truth Aligned)

All metadata configurations were evaluated on an identical test set of 1 million synthetic questions. Each question was generated directly from a target chunk, enabling exact ground-truth evaluation.

Recall@K measures whether the correct chunk appears within the top-K retrieved results. MRR measures the rank position of the first correct retrieval.

|   Test |   Recall@10 |   Recall@100 |   Recall@1000 |     MRR |
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
|     10 |     **0.30815** |      **0.40006** |       **0.4692**  | **0.21582** |
|     11 |     0.29164 |      0.37642 |       0.44017 | 0.20812 |
|     12 |     0.30531 |      0.39581 |       0.46299 | 0.21504 |
|     13 |     0.30223 |      0.39513 |       0.46393 | 0.21077 |
|     14 |     0.30112 |      0.39273 |       0.4612  | 0.21243 |
|     15 |     0.3028  |      0.39383 |       0.46259 | 0.21092 |

---

## 2. Reworded Synthetic Queries (LLM-Paraphrased)

The same evaluation pipeline was used as in the direct synthetic query experiment. The only difference is that each synthetic query was additionally rephrased using an LLM to introduce lexical and syntactic variation.

This reduces direct wording overlap between queries and target chunks, producing a more realistic approximation of natural user queries.

|   Test |   Recall@10 |   Recall@100 |   Recall@1000 |     MRR |
|-------:|------------:|-------------:|--------------:|--------:|
|      1 |     0.26715 |      0.35626 |       0.42712 | 0.18797 |
|      2 |     0.26675 |      0.35452 |       0.42389 | 0.18798 |
|      3 |     0.28841 |      0.37993 |       0.44994 | 0.19865 |
|      4 |     0.28069 |      0.37045 |       0.43982 | 0.19917 |
|      5 |     0.26949 |      0.35762 |       0.42736 | 0.18845 |
|      6 |     0.2785  |      0.36251 |       0.42875 | 0.1975  |
|      7 |     0.27807 |      0.36619 |       0.4335  | 0.19432 |
|      8 |     0.28324 |      0.37476 |       0.44568 | 0.19653 |
|      9 |     0.27609 |      0.36467 |       0.43452 | 0.19524 |
|     10 |     0.28749 |      0.37901 |       0.44999 | 0.19873 |
|     11 |     0.27174 |      0.35782 |       0.42292 | 0.19181 |
|     12 |     0.28514 |      0.37486 |       0.4437  | 0.19828 |
|     13 |     0.28206 |      0.3748  |       0.44572 | 0.19438 |
|     14 |     0.28135 |      0.37134 |       0.44145 | 0.19582 |
|     15 |     0.28267 |      0.37427 |       0.4438  | 0.19475 |

---

# Pipeline Overview

1_raw.py
→ 2_sections.py
→ 3_chunks.py
→ 4_metadata.py
→ 5_embeddings.py / 5_embeddings_vllm.py
→ 6_faiss.py
→ 7_samples.py
→ 8_questions.py / 8_questions_vllm.py
→ 9_refine_vllm.py
→ 10_reword_vllm.py
→ 11_embed_questions.py
→ 12_test.py

Each stage produces artifacts used downstream for embedding and retrieval experiments.

---

# Data Pipeline Stages

## 1. Raw Wikipedia Extraction (`1_raw.py`)

Parses a compressed Wikipedia XML dump (`bz2`) and extracts article content from the main namespace.

**Key features:**
- Streaming XML parsing via `lxml.iterparse`
- Filters:
  - non-article namespaces
  - redirects
  - disambiguation pages

Output: raw_batches/batch_<n>.jsonl

Each record contains:
- `page_title`
- `page_id`
- raw article text

---

## 2. Section Segmentation (`2_sections.py`)

Converts raw article text into structured sections using wiki markup parsing.

**Key features:**
- Uses `mwparserfromhell`
- Removes:
  - References
  - External links
  - boilerplate sections
  - Preserves:
  - section hierarchy
  - section paths
  - section depth

Output: section_batches/*.jsonl


Each record includes:
- page metadata
- section title
- hierarchy path
- cleaned section text
- wikilinks

---

## 3. Chunking (`3_chunks.py`)

Splits sections into embedding-friendly chunks.

**Key features:**
- Chunk size: ~800–1600 characters
- Sentence-aware splitting using NLTK
- Preserves section metadata

Output: chunk_parquets/*.parquet


Each chunk includes:
- page + section metadata
- hierarchical path
- chunk text
- lead sentence (when applicable)

---

## 4. Metadata Augmentation (`4_metadata.py`)

Experiments with injecting structured metadata into chunk text.

**Key idea:**
Prefix chunks with structured context such as:
- page title
- section title
- hierarchy depth
- section position
- section path (optional)

Two configurations:
- `NEW_FIELDS`
- `FULL_FIELDS`

Output: chunk_augmented_<test_id>/*.parquet


Each chunk becomes: \[metadata prefix\] + original text


---

## 5. Embedding Generation

### Standard (`5_embeddings.py`)

- Uses `SentenceTransformer`
- Model: `BAAI/bge-small-en-v1.5`
- Single GPU per process
- Normalized embeddings

---

### vLLM Multi-GPU (`5_embeddings_vllm.py`)

- Distributed embedding generation
- One process per GPU
- Higher throughput via VLLM backend
- Token truncation: 512 tokens

Output: embeddings_<test_id>/*.parquet


Each row:
- `id`
- embedding vector

---

## 6. Vector Indexing (`6_faiss.py`)

Builds a FAISS Approximate Nearest Neighbor index.

**Key features:**
- IVF-PQ index
- Cosine similarity (inner product + normalization)
- Training sample: 1.5M vectors
- Batch insertion at scale

Output: "faiss_indexes"/ivfpq_index_<test_id>.faiss


---

## 7. Sampling (`7_samples.py`)

Creates a fixed evaluation subset of Wikipedia chunks.

- Random sample: 1,000,000 chunks
- Preserves global IDs
- Efficient file-level selection

Output: samples.parquet


---

## 8. Synthetic Question Generation

### Standard (`8_questions.py`)
Uses:
- `mistralai/Mistral-7B-v0.1`

Generates search-style questions from passages.

---

### vLLM (`8_questions_vllm.py`)

- Batch inference
- Higher throughput
- Includes validation:
  - single sentence constraint
  - question formatting checks

Output: question_parquets/*.parquet


---

## 9. Question Refinement (`9_refine_vllm.py`)

- Detects invalid questions
- Regenerates only bad samples
- Iterates until convergence

---

## 10. Question Rewording (`10_reword_vllm.py`)

Rewrites questions while preserving meaning.

Checks:
- single sentence
- valid question format
- no metadata leakage

---

## 11. Question Embeddings (`11_embed_questions.py`)

Encodes questions into vector space.

- Optional reworded embeddings
- GPU-specific execution
- Model: `BAAI/bge-small-en-v1.5`

Output: question_embeddings/ or reworded_embeddings/


---

## 12. Retrieval Evaluation (`12_test.py`)

Evaluates FAISS retrieval performance.

**Metrics:**
- Recall@10
- Recall@100
- Recall@1000
- MRR (Mean Reciprocal Rank)

Process:
1. Embed question
2. Query FAISS index
3. Check if correct chunk ID is retrieved

---

# Key Research Focus

- Chunking strategy impact on retrieval
- Effect of metadata injection into embeddings
- Synthetic query quality and diversity
- Embedding robustness under structured vs raw text
- Large-scale FAISS retrieval performance

---

# Tech Stack

- Python
- Wikipedia XML dumps
- lxml, mwparserfromhell
- PyArrow / Parquet
- SentenceTransformers
- VLLM (Mistral-7B)
- FAISS (IVF-PQ)
- NLTK

---

# Output Artifacts

- Raw Wikipedia batches (JSONL)
- Sectioned data (JSONL)
- Chunked datasets (Parquet)
- Metadata-augmented chunks
- Embeddings (chunk + question)
- FAISS index
- Synthetic question datasets
- Retrieval evaluation metrics

---

# Summary

This project implements a full pipeline for evaluating how chunking strategy, metadata augmentation, and synthetic query generation affect embedding-based retrieval systems at Wikipedia scale.

