# 4_metadata.py


import os
import pandas as pd


# Test configurations to generate augmented chunk outputs
TESTS = [4, 5, 6, 7, 9, 11, 15]

# Source parquet files containing chunked text
INPUT_DIR = "chunk_parquets"

# Maximum metadata length in characters to prepend
MAX_META_LEN = 200

NEW_FIELDS = {
    1: [],
    2: ["path"],

    3: ["page"],
    4: ["section"],
    5: ["parent"],
    6: ["level"],
    7: ["pos"],

    8: ["page", "section"],
    9: ["parent", "section"],
    10: ["page", "parent", "section"],

    11: ["level", "pos"],
    12: ["page", "section", "level"],
    13: ["page", "section", "pos"],
    14: ["page", "section", "level", "pos"],

    15: ["page", "parent", "section", "level", "pos"],
}

# Full metadata field combinations for alternate experiments
FULL_FIELDS = {
    1: [],
    2: ["path"],

    3: ["page"],
    4: ["section"],
    5: ["parent"],

    6: ["page", "parent"],
    7: ["page", "section"],
    8: ["parent", "section"],
    9: ["page", "parent", "section"],

    10: ["level"],
    11: ["pos"],

    12: ["level", "pos"],

    13: ["page", "level"],
    14: ["page", "pos"],
    15: ["page", "level", "pos"],

    16: ["section", "level"],
    17: ["section", "pos"],
    18: ["section", "level", "pos"],

    19: ["page", "section", "level"],
    20: ["page", "section",  "pos"],
    21: ["page", "section", "level", "pos"],

    22: ["page", "parent", "section", "level"],
    23: ["page", "parent", "section",  "pos"],
    24: ["page", "parent", "section", "level", "pos"],
}


def build_input(record, fields):
    # Build a metadata prefix from selected fields and append the original text
    meta_lines = []

    for f in fields:
        if f == "page" and record["page"]:
            meta_lines.append(f"page: {record['page']}")
        elif f == "parent" and record["parent"]:
            meta_lines.append(f"parent: {record['parent']}")
        elif f == "section" and record["section"]:
            meta_lines.append(f"section: {record['section']}")
        elif f == "level":
            meta_lines.append(f"depth: {record['level']}")
        elif f == "pos":
            meta_lines.append(f"position: {record['pos']}")

    meta = "\n".join(meta_lines)

    if "path" in fields and record["path"]:
            # Append truncated path metadata without exceeding budget
            if meta:
                meta += "\npath: "
            else:
                meta += "path: "

            remaining_len = MAX_META_LEN - len(meta)

            parts = record["path"].split("\\/")
            path = ""

            for part in reversed(parts):
                if len(path) + len(part) + 1 > remaining_len:
                    break
                path = part + ">" + path

            if path and path.endswith(">"):
                path = path[:-1]
            else:
                meta += path

    # Enforce maximum metadata length before appending text
    if len(meta) > MAX_META_LEN:
        meta = meta[:MAX_META_LEN]

    return meta + "\n\n" + record["text"]


def gen_test(test_num):
    # Create output directory for this test configuration
    output_dir = f"chunk_augmented_{test_num}"
    os.makedirs(output_dir, exist_ok=True)

    fields = NEW_FIELDS.get(test_num, [])

    for fname in os.listdir(INPUT_DIR):
        if not fname.endswith(".parquet"):
            continue

        path = os.path.join(INPUT_DIR, fname)
        df = pd.read_parquet(path)

        # Prepend selected metadata fields to each chunk's text
        new_texts = []
        for row in df.itertuples(index=False):
            record = row._asdict()
            new_texts.append(build_input(record, fields))

        df["text"] = new_texts

        out_path = os.path.join(output_dir, fname)
        df.to_parquet(out_path, index=False)

        print(f"processed {fname}")


def main():
    # Generate augmented outputs for each selected test configuration
    for num in TESTS:
        gen_test(num)
        

if __name__ == "__main__":
    main()