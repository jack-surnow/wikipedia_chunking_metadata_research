# Wikipedia Chunking & Metadata Research

This repository contains an end-to-end experimental pipeline for investigating how Wikipedia text chunking and metadata enrichment affect embedding quality, retrieval performance, and synthetic question generation for RAG-style systems.

The project builds a full pipeline from raw Wikipedia XML → structured chunks → metadata augmentation → embeddings → FAISS indexing → synthetic question generation → retrieval evaluation.

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

