# 2_sections.py


import mwparserfromhell
import json
import os
import re


# Input and output directories
INPUT_DIR = "raw_batches"
OUTPUT_DIR = "section_batches"

# Starting batch number for processing (allows resuming from checkpoints)
START_BATCH_NUMBER = 0

# Number of pages per batch in output files
PAGES_PER_BATCH = 1000

# Text length constraints for sections
MAX_TEXT_LENGTH = 500
MIN_TEXT_LENGTH = 100

# Filter out non-content sections like References, External Links, etc.
REMOVED_SECTIONS = r'^(references?|external links?|see also|further reading|bibliography|notes?|sources?|works cited|footnotes?|citations?|references and notes|notes and references|general bibliography|explanatory notes)$'
COMPILED_REMOVED_SECTIONS = re.compile(REMOVED_SECTIONS, re.IGNORECASE)

# Filter out file/image wikilinks
REMOVED_WIKILINKS = r'^(file|image|audio|video|media):'
COMPILED_REMOVED_WIKILINKS = re.compile(REMOVED_WIKILINKS, re.IGNORECASE)

# Filter out template boilerplate like Infobox, Navbox, Citations, etc.
REMOVED_TEMPLATES = r'^(infobox|navbox|sidebar|citation|cite|footnote|reflist|disambiguation|authority control|as of|dead link|multiple issues|cleanup|unreferenced|sfn|sfnm|sfnr|refn|multiple image|short description|coord|portal|wiktionary|hatnote|main|orphan|dead end)'
COMPILED_REMOVED_TEMPLATES = re.compile(REMOVED_TEMPLATES, re.IGNORECASE)

# Filter out markup tags that don't contribute to main content
REMOVED_TAGS = r'^(ref|references|gallery|timeline|score|imagemap|syntaxhighlight)$'
COMPILED_REMOVED_TAGS = re.compile(REMOVED_TAGS, re.IGNORECASE)

# Identify redirect pages to skip
COMPILED_REDIRECT = re.compile(r'^\s*#redirect\s*\[\[', re.IGNORECASE)

# Identify disambiguation pages to skip
COMPILED_DISAMBIGUATION = re.compile(r'(?:\(disambiguation\)|\(disambiguation page\))$', re.IGNORECASE)

# Type aliases for mwparserfromhell nodes
Wikilink = mwparserfromhell.nodes.wikilink.Wikilink
Template = mwparserfromhell.nodes.template.Template
Heading = mwparserfromhell.nodes.heading.Heading
External_Link = mwparserfromhell.nodes.external_link.ExternalLink
Comment = mwparserfromhell.nodes.comment.Comment
Tag = mwparserfromhell.nodes.tag.Tag
Argument = mwparserfromhell.nodes.argument.Argument
Html_Entity = mwparserfromhell.nodes.html_entity.HTMLEntity
Text = mwparserfromhell.nodes.text.Text


def count_headings(wikicode):
    # Count total number of headings in wikicode
    return len(wikicode.filter_headings())


def extract_wikilinks(wikicode):
    # Extract unique wikilink targets from wikicode
    wikilinks = set()
    
    for link in wikicode.filter_wikilinks(recursive=False):
        if not link.title:
            continue

        target = link.title.strip_code().strip()
        if not target:
            continue

        wikilinks.add(target)

    return list(wikilinks)


def remove_headings(wikicode):
    # Remove all heading nodes from wikicode
    for node in reversed(wikicode.filter_headings()):
        wikicode.remove(node)


def clean_wikicode(wikicode):
    # Recursively clean wikicode by removing comments, arguments, and unwanted templates/tags
    cleaned_nodes = []

    for node in wikicode.nodes:
        node_type = type(node)

        # Skip comment nodes
        if node_type is Comment:
            continue

        # Skip argument nodes
        if node_type is Argument:
            continue

        # Keep heading nodes
        if node_type is Heading:
            cleaned_nodes.append(node)
            continue

        # Convert HTML entities to text
        if node_type is Html_Entity:
            cleaned_nodes.append(Text(node.normalize()))
            continue

        # Extract text from external links
        if node_type is External_Link:
            if node.title:
                cleaned_nodes.append(Text(str(node.title)))
            continue

        # Keep non-empty text nodes
        if node_type is Text:
            if node.value.strip():
                cleaned_nodes.append(node)
            continue

        # Filter wikilinks but keep valid ones
        if node_type is Wikilink:
            if not COMPILED_REMOVED_WIKILINKS.match(re.sub(r'\s+', ' ', node.title.strip()).casefold()):
                cleaned_nodes.append(node)
            continue
            
        # Filter and recursively clean templates
        elif node_type is Template:
            if COMPILED_REMOVED_TEMPLATES.match(re.sub(r'\s+', ' ', node.name.strip()).casefold()):
                continue

            # Recursively clean template parameters
            for param in node.params:
                if param.value:
                    cleaned_value = clean_wikicode(param.value)
                    param.value = cleaned_value

            cleaned_nodes.append(node)
            continue

        # Filter and recursively clean tags
        elif node_type is Tag:
            if COMPILED_REMOVED_TAGS.match(node.tag.strip().casefold()):
                continue

            # Recursively clean the contents inside the tag
            if node.contents:
                cleaned_nodes.extend(clean_wikicode(node.contents).nodes)
            continue

        # Keep all other nodes
        cleaned_nodes.append(node)

    return mwparserfromhell.wikicode.Wikicode(cleaned_nodes)


def save_first_paragraph(text):
    # Extract first paragraph (up to double newline or single newline)
    idx = text.find("\n\n")
    if idx != -1:
        return text[:idx].strip()

    idx = text.find("\n")
    if idx != -1:
        return text[:idx].strip()

    return text.strip()


def extract_sections(page_title, wiki_text) -> list:
    # Parse wiki text and extract sections with hierarchical structure
    wikicode = clean_wikicode(mwparserfromhell.parse(wiki_text))

    skipped_level = -1
    sections = []
    last_at_levels = [page_title]

    section_count = 0
    number_of_headings = count_headings(wikicode)

    for section in wikicode.get_sections(flat=True, include_lead=True, include_headings=True):
        # Get heading from section
        heading = section.filter_headings()
        section_count += 1

        # If there's a heading, use it as the section title
        if heading:
            section_title = heading[0].title.strip_code().strip()
            level = heading[0].level - 1

            # Skip unwanted sections and track skipping level
            if COMPILED_REMOVED_SECTIONS.match(re.sub(r'\s+', ' ', section_title).casefold()):
                if skipped_level == -1 or level < skipped_level:
                    skipped_level = level
                continue
            else:
                # Skip subsections of removed sections
                if skipped_level != -1 and level > skipped_level:
                    continue
                else:
                    skipped_level = -1

            # Update section hierarchy tracking
            if len(last_at_levels) < level:
                last_at_levels += [""] * (level - len(last_at_levels))
            last_at_levels = last_at_levels[:level] + [section_title]

            if level >= 2:
                parent_title = last_at_levels[level - 1]
            else:
                parent_title = ""
        else:
            # Lead section (introduction before first heading)
            section_title = "(Lead)"
            level = 1
            skipped_level = -1
            last_at_levels = [page_title, section_title]
            parent_title = ""

        # Build hierarchical path
        section_path = "\\/".join(last_at_levels)

        # Extract wikilinks from section
        wikilinks = list(extract_wikilinks(section))

        # Extract and clean plain text
        remove_headings(section)
        text = section.strip_code().strip()

        # Calculate position in document (0.0-1.0)
        position = round(section_count / number_of_headings, 2) if number_of_headings else 0

        # Only include sections with sufficient text
        if text and len(text) >= MIN_TEXT_LENGTH:
            sections.append({
                "page": page_title,
                "section": section_title,
                "parent": parent_title,
                "path": section_path,
                "level": level,
                "pos": position,
                "wikilinks": wikilinks,
                "text": text
            })

    return sections


def iter_jsonl(path):
    # Yield each JSON object from JSONL file
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def save_pages(output_path, pages):
    # Write sections from pages to JSONL, with each section as separate record
    with open(output_path, "w", encoding="utf-8") as out:
        page_id = 0

        for page in pages:
            section_id = 0
            for section in page:
                record = {
                    "pid": page_id,
                    "sid": section_id,
                    "page": section["page"],
                    "parent": section["parent"],
                    "section": section["section"],
                    "path": section["path"],
                    "level": section["level"],
                    "pos": section["pos"],
                    "wikilinks": section["wikilinks"],
                    "text": section["text"]
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                section_id += 1

            page_id += 1


def main():
    # Process input batches to extract sections and save to output batches
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Iterate through each input file
    for filename in sorted(os.listdir(INPUT_DIR)):
        if not filename.endswith(".jsonl"):
            continue

        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)

        pages = []

        # Extract sections from each page
        for page in iter_jsonl(input_path):

            title = page["page_title"]
            raw_text = page["raw"]

            # Extract sections from this page
            pages.append(extract_sections(title, raw_text))

        # Save all sections to output file
        save_pages(output_path, pages)


if __name__ == "__main__":
    main()