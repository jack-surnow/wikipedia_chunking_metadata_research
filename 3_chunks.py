# 3_chunks.py


import os
import json
import pyarrow as pa
import pyarrow.parquet as pq
import nltk

nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from nltk.tokenize import PunktSentenceTokenizer


INPUT_DIR = "section_batches"
OUTPUT_DIR = "chunk_parquets"

# Number of chunk records to buffer before writing a parquet batch
BATCH_SIZE = 100_000

# Chunk size parameters in characters
MIN_CHUNK_SIZE = 800
MAX_CHUNK_SIZE = 1600
OVERLAP = 100

# Sentence tokenizer for splitting on sentence boundaries
TOKENIZER = PunktSentenceTokenizer()


def chunk_text(text):
    # Split text into chunks that are roughly bounded by paragraphs, newlines, or sentences
    chunks = []
    i = 0
    n = len(text)

    # Precompute sentence spans once for sentence-level split decisions
    sentence_spans = list(TOKENIZER.span_tokenize(text))
    m = len(sentence_spans)
    sent_idx = 0

    while i < n:
        if n - i <= MAX_CHUNK_SIZE:
            chunk = text[i:n].strip()
            if chunk:
                chunks.append(chunk)
            break

        split_idx = -1
        overlap_split = False

        limit = i + MAX_CHUNK_SIZE
        min_pos = i + MIN_CHUNK_SIZE

        window = text[i:limit]

        # Priority 1: split at paragraph boundary if it falls within allowed range
        para_idx = window.rfind("\n\n")
        if para_idx != -1:
            para_idx += 2
            if para_idx >= MIN_CHUNK_SIZE:
                split_idx = para_idx

        # Priority 2: split at the last newline if paragraph split not available
        if split_idx == -1:
            nl_idx = window.rfind("\n")
            if nl_idx != -1:
                nl_idx += 1
                if nl_idx >= MIN_CHUNK_SIZE:
                    split_idx = nl_idx

        # Priority 3: split at the last sentence boundary before the max chunk size
        if split_idx == -1:
            # Advance to the first sentence that starts after the current chunk begin
            while sent_idx < m and sentence_spans[sent_idx][1] <= i:
                sent_idx += 1

            j = sent_idx
            best_end = -1

            while j < m:
                end = sentence_spans[j][1]

                if end > limit:
                    break

                if end >= min_pos:
                    best_end = end

                j += 1

            if best_end != -1:
                split_idx = best_end - i

        # Fallback: if no natural split found, split at max chunk size with overlap
        if split_idx == -1:
            split_idx = MAX_CHUNK_SIZE
            overlap_split = True

        # Emit chunk text for current window
        chunk = text[i:i + split_idx].strip()
        if chunk:
            chunks.append(chunk)

        # Advance pointer, using overlap only for fallback splits
        if overlap_split:
            i += split_idx - OVERLAP
        else:
            i += split_idx

    return chunks


def convert_pos(pos: float) -> str:
    if pos < 0.33:
        return "start"
    elif pos < 0.67:
        return "mid"
    else:
        return "end"
    

def get_first_sentence(text):
    sentences = sent_tokenize(text)
    return sentences[0] if sentences else ""


def main():
    # Parse section batches, chunk text, and write out parquet files
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rows = []
    chunk_id = 0
    batch_num = 0

    current_page = ""
    first_sentence = ""

    # Iterate through all section batch files
    for fname in os.listdir(INPUT_DIR):
        if not fname.endswith(".jsonl"):
            continue

        print(f"Reading: {fname}")

        input_path = os.path.join(INPUT_DIR, fname)

        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if record["page"] != current_page:
                    current_page = record["page"]
                    first_sentence = ""

                # Preserve first sentence from the lead section as context for later chunks
                if record["section"] == "(Lead)":
                    first_sentence = get_first_sentence(record["text"])

                chunks = chunk_text(record["text"])

                for chunk in chunks:
                    new_record = {
                        "id": chunk_id,
                        "page": record["page"],
                        "parent": record["parent"],
                        "section": record["section"],
                        "path": record["path"],
                        "level": record["level"],
                        "pos": convert_pos(record["pos"]),
                        "first_sentence": first_sentence,
                        "text": chunk,
                    }

                    rows.append(new_record)
                    chunk_id += 1

                    # Flush to parquet once batch size is reached
                    if len(rows) >= BATCH_SIZE:
                        table = pa.Table.from_pylist(rows)

                        out_path = os.path.join(OUTPUT_DIR, f"chunk_batch_{batch_num}.parquet")

                        writer = pq.ParquetWriter(out_path, table.schema)
                        writer.write_table(table)
                        writer.close()

                        batch_num += 1
                        rows.clear()

    # Flush any remaining chunk rows to a final parquet batch
    if rows:
        table = pa.Table.from_pylist(rows)

        out_path = os.path.join(OUTPUT_DIR, f"chunk_batch_{batch_num}.parquet")

        writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        writer.close()


if __name__ == "__main__":
    main()