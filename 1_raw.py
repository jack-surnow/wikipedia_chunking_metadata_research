# obtain_raw.py


import bz2
from lxml import etree
import json
import os
import re


# Configuration for Wikipedia data extraction
OUTPUT_DIR = "raw_batches"

# Starting batch number for processing (allows resuming from checkpoints)
START_BATCH_NUMBER = 0

# Number of pages to include in each output batch
PAGES_PER_BATCH = 1000

# Identify redirect pages to skip
COMPILED_REDIRECT = re.compile(r'^\s*#redirect\s*\[\[', re.IGNORECASE)

# Identify disambiguation pages to skip
COMPILED_DISAMBIGUATION = re.compile(r'(?:\(disambiguation\)|\(disambiguation page\))$', re.IGNORECASE)


def save_pages(batch_number, pages):
    # Write batch of pages to JSONL file with page metadata
    with open(f"{OUTPUT_DIR}/batch_{batch_number}.jsonl", "w", encoding="utf-8") as out:
        page_id = 0

        for page in pages:
            record = {
                "page_title": page["page_title"],
                "page_id": page_id,
                "raw": page["raw"]
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            page_id += 1


def main():
    # Parse Wikipedia XML dump and extract raw article text by namespace
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    unknown_count = 0
    # Stream parse Wikipedia XML to avoid loading entire dump in memory
    with bz2.open("C:/Wikipedia Datasets/enwiki-20260301-pages-articles-multistream.xml.bz2", "rb") as f:
        context = etree.iterparse(f, events=("end",), tag="{*}page")

        batch_number = 0

        pages = []
        page_count = 0

        for event, elem in context:
            # Free memory by removing processed elements
            while elem.getprevious() is not None:
                del elem.getparent()[0]

            # Only process main namespace (0), skip talk pages and other namespaces
            ns = elem.findtext(".//{*}ns")
            if ns != "0":
                elem.clear()
                continue

            # Extract page title
            title_elem = elem.find(".//{*}title")
            if (title_elem is not None):
                title = title_elem.text
            else:
                title = f"Unknown-{unknown_count}"
                unknown_count += 1

            # Extract page text content
            text_elem = elem.find(".//{*}text")
            if text_elem is None or not text_elem.text:
                elem.clear()
                continue

            raw_text = text_elem.text

            # Skip redirects and disambiguation pages
            if COMPILED_REDIRECT.match(raw_text.lstrip()) or COMPILED_DISAMBIGUATION.match(title):
                elem.clear()
                continue

            # Skip pages before START_BATCH_NUMBER for resume capability
            if batch_number < START_BATCH_NUMBER:
                page_count += 1
                if page_count >= PAGES_PER_BATCH:
                    page_count = 0
                    batch_number += 1

                elem.clear()
                continue

            pages.append({
                "page_title": title,
                "raw": raw_text
            })

            page_count += 1

            # Save batch when reaching page limit
            if page_count >= PAGES_PER_BATCH:
                save_pages(batch_number, pages)
                pages = []
                page_count = 0
                batch_number += 1

            # Testing
            # if batch_number >= 2:
            #     break

            elem.clear()

        # Save final partial batch
        save_pages(batch_number, pages)


if __name__ == "__main__":
    main()