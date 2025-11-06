#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate JSON and Markdown files for the SICK dataset from parsed XML files.

Usage:
  Run this script directly inside the Docker container or local environment:

      python gen_json_sick.py [--project-root /app] [--out-dir /app/json]

Inputs:
  - Parsed XML files under the ccg2lambda *app* root (in Docker this is /app):

        en/parsed/sick_{test|train}_*.txt.sem.xml

    Each XML file must contain two <sentence> elements:
      * The first (<sentence id="0">) is the premise.
      * The second (<sentence id="1">) is the hypothesis.

    The logical formulas for each sentence are found in its <semantics> block:
      * Premise: span id="s0_sp0"
      * Hypothesis: span id="s1_sp0"

  - NLI gold labels are not embedded in XML files. They are stored separately as:

        en/plain/sick_{test|train}_N.answer

    Example mapping:
        sick_test_6.txt.sem.xml  →  sick_test_6.txt  →  en/plain/sick_test_6.answer

    The first non-empty line of the corresponding .answer file is interpreted as the gold label.

Outputs:
  - JSON and Markdown files written to the output directory (default: ./json or /app/json):

        json/sick_train.json
        json/sick_train.md
        json/sick_test.json
        json/sick_test.md

    You can override the output directory with the --out-dir argument:

        python gen_json_sick.py --out-dir /custom/path/to/output

Each JSON entry contains the following fields:
  - ID                        (e.g., sick_train_1234.txt)
  - premise                   (surface text)
  - hypothesis                (surface text)
  - NLI-gold-label            (yes / no / unknown / None)
  - premise-logical-formulas  (raw logical formula string)
  - hypothesis-logical-formulas (raw logical formula string)

Notes:
  - If the corresponding .answer file is missing or unreadable, the label will be recorded as "None".
  - Markdown output is formatted with LaTeX-style math blocks for proper rendering in HackMD or MathJax.
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional


def extract_sentence_text(sentence_elem: ET.Element) -> str:
    """Reconstruct sentence text from the <tokens>/<token> sequence."""
    tokens_elem = sentence_elem.find("tokens")
    if tokens_elem is None:
        return ""

    surfs: List[str] = []
    for tok in tokens_elem.findall("token"):
        surf = tok.get("surf", "")
        if surf:
            surfs.append(surf)

    # Simple join with spaces; this is good enough for our purposes.
    return " ".join(surfs)


def extract_logical_formula(root: ET.Element, span_id: str) -> str:
    """
    From the <semantics> section, extract the logical formula contained in
    the span with a given id (e.g., "s0_sp0", "s1_sp0").

    In the SICK *.sem.xml files produced by ccg2lambda, each sentence has its
    own <semantics> block, and inside that block the logical formulas are
    usually stored in the "sem" attribute of <span> elements.
    """
    semantics_sections: List[ET.Element] = []
    # Collect all <semantics> sections (ignoring namespaces), because each
    # sentence has its own <semantics> block, and both may contain spans
    # with ids like "s0_sp0", "s1_sp0", etc.
    for elem in root.iter():
        if elem.tag.split("}")[-1] == "semantics":
            semantics_sections.append(elem)

    if not semantics_sections:
        return ""

    target_span: Optional[ET.Element] = None
    # Search only inside <semantics> blocks to avoid picking up the CCG
    # <span> elements (which also have ids like "s0_sp0" but no sem attribute).
    for semantics in semantics_sections:
        for span in semantics.iter():
            if span.tag.split("}")[-1] == "span" and span.get("id") == span_id:
                target_span = span
                break
        if target_span is not None:
            break

    if target_span is None:
        return ""

    # First, try to read the logical form from the "sem" *attribute* on the span
    # (this is how ccg2lambda encodes formulas in sick_*.txt.sem.xml).
    attr_sem = target_span.get("sem")
    if attr_sem:
        return attr_sem.strip()

    # Fallback: look for a <sem> child element (ignoring namespaces).
    sem_elem: Optional[ET.Element] = None
    for child in target_span.iter():
        if child.tag.split("}")[-1] == "sem":
            sem_elem = child
            break

    if sem_elem is None or sem_elem.text is None:
        return ""

    return sem_elem.text.strip()


def get_nli_label(xml_path: Path, pair_id: str) -> str:
    """Look up the NLI gold label for a pair using the corresponding .answer file.

    For SICK in this project, the gold label is stored in:

        en/plain/sick_{train|test}_N.answer

    where N matches the number in the XML/ID, e.g.:

        sick_test_6.txt.sem.xml  ->  sick_test_6.txt  ->  sick_test_6.answer

    This function derives the .answer file path from the XML path and
    pair_id, reads the first non-empty line, and normalizes a few common
    label spellings.
    """
    # xml_path is typically .../en/parsed/sick_*.txt.sem.xml
    # We want the sibling directory .../en/plain/.
    try:
        en_dir = xml_path.parents[1]  # .../en
    except IndexError:
        return "None"

    plain_dir = en_dir / "plain"

    # pair_id is usually like "sick_train_1.txt"; strip the .txt to get
    # the base and add the .answer extension.
    base = pair_id[:-4] if pair_id.endswith(".txt") else pair_id
    answer_path = plain_dir / f"{base}.answer"

    if not answer_path.exists():
        return "None"

    try:
        with answer_path.open("r", encoding="utf-8") as f:
            for line in f:
                label = line.strip()
                if not label:
                    continue
                return label
    except Exception:
        return "None"

    return "None"


def xml_to_entry(xml_path: Path) -> Dict[str, str]:
    """
    Convert one XML file (containing a premise/hypothesis pair) into
    a single JSON-ready dict with the required keys.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Derive an ID from the file name, e.g.:
    #   sick_train_1234.txt.sem.xml -> sick_train_1234.txt
    fname = xml_path.name
    # Prefer to strip the trailing '.sem.xml' and keep the '.txt' part,
    # e.g. 'sick_train_1.txt.sem.xml' -> 'sick_train_1.txt'.
    if fname.endswith(".sem.xml"):
        pair_id = fname[:-len(".sem.xml")]
    elif fname.endswith(".xml"):
        pair_id = fname[:-4]  # generic fallback
    else:
        pair_id = fname

    # Navigate to the two sentences
    sentence_elems = root.findall(".//sentences/sentence")
    if len(sentence_elems) < 2:
        raise ValueError(
            f"Expected at least 2 <sentence> elements in {xml_path}, found {len(sentence_elems)}"
        )
    premise_sent = sentence_elems[0]
    hypothesis_sent = sentence_elems[1]

    premise_text = extract_sentence_text(premise_sent)
    hypothesis_text = extract_sentence_text(hypothesis_sent)

    # For the first sentence, the root span id is typically "s0_sp0"
    # and for the second "s1_sp0".
    premise_formula = extract_logical_formula(root, "s0_sp0")
    hypothesis_formula = extract_logical_formula(root, "s1_sp0")

    entry: Dict[str, str] = {
        "ID": pair_id,
        "premise": premise_text,
        "hypothesis": hypothesis_text,
        "NLI-gold-label": get_nli_label(xml_path, pair_id),
        "premise-logical-formulas": premise_formula,
        "hypothesis-logical-formulas": hypothesis_formula,
    }
    return entry


def collect_entries_for_split(project_root: Path, split: str) -> List[Dict[str, str]]:
    """
    Collect all XML files for a given split ('train' or 'test')
    and convert them into JSON entries.
    """
    parsed_dir = project_root / "en" / "parsed"
    pattern = f"sick_{split}_*.txt.sem.xml"
    xml_files = sorted(parsed_dir.glob(pattern))

    if not xml_files:
        print(f"[WARN] No XML files matching {pattern} under {parsed_dir}", file=sys.stderr)

    entries: List[Dict[str, str]] = []
    for xml_path in xml_files:
        try:
            entry = xml_to_entry(xml_path)
            entries.append(entry)
        except Exception as e:
            print(f"[ERROR] Failed to process {xml_path}: {e}", file=sys.stderr)

    return entries



def write_json(entries: List[Dict[str, str]], out_path: Path) -> None:
    """Write a list of dicts to a JSON file (UTF-8, pretty-printed)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Wrote {len(entries)} entries to {out_path}")


def to_latex_formula(formula: str) -> str:
    """Convert a raw logical formula string into a LaTeX-friendly one for HackMD/MathJax.

    This performs a few lightweight textual rewrites so that common logical
    symbols render nicely (e.g., exists -> \\exists, & -> \\land).
    The transformation is intentionally conservative to avoid changing the
    structure of the formula.
    """
    # Escape backslashes just in case (there are usually none in the input).
    s = formula.replace("\\", "\\\\")

    # Quantifiers: exists, forall -> LaTeX commands.
    s = re.sub(r"\bexists\b", r"\\exists", s)
    s = re.sub(r"\bforall\b", r"\\forall", s)

    # Logical connectives.
    s = s.replace("->", r" \\rightarrow ")
    s = s.replace("&", r" \\land ")

    return s


# Write entries as Markdown (for manual inspection)
def write_markdown(entries: List[Dict[str, str]], md_path: Path) -> None:
    """Write entries into a Markdown file suitable for manual inspection (e.g., on HackMD)."""
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# SICK dataset entries — {}\n\n".format(md_path.stem))
        for entry in entries:
            f.write("## ID: {}\n\n".format(entry["ID"]))
            f.write("**Premise**:\n\n{}\n\n".format(entry["premise"]))
            f.write("**Hypothesis**:\n\n{}\n\n".format(entry["hypothesis"]))
            f.write("**Gold-label**: {}\n\n".format(entry["NLI-gold-label"]))

            premise_formula = entry.get("premise-logical-formulas", "")
            hypothesis_formula = entry.get("hypothesis-logical-formulas", "")

            if premise_formula:
                f.write("**Premise-logical-formulas:**\n\n")
                latex_premise = to_latex_formula(premise_formula)
                f.write("$$\n{}\n$$\n\n".format(latex_premise))
            else:
                f.write("**Premise-logical-formulas:** –\n\n")

            if hypothesis_formula:
                f.write("**Hypothesis-logical-formulas:**\n\n")
                latex_hyp = to_latex_formula(hypothesis_formula)
                f.write("$$\n{}\n$$\n\n".format(latex_hyp))
            else:
                f.write("**Hypothesis-logical-formulas:** –\n\n")
    print(f"[INFO] Wrote Markdown file to {md_path}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate SICK JSON files from parsed XML (ccg2lambda)."
    )
    script_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help=(
            "Path to the ccg2lambda *app* root. Inside Docker this is typically /app. On the host it is usually /path/to/ccg2lambda. Default: inferred as the directory one level above this script."
        ),
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="*",
        default=["train", "test"],
        help='Which splits to process (default: "train test").',
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for JSON/Markdown files (default: PROJECT_ROOT/json).",
    )

    args = parser.parse_args(argv)

    project_root = (
        Path(args.project_root).resolve()
        if args.project_root is not None
        else script_root
    )

    if args.out_dir is not None:
        out_dir = Path(args.out_dir).resolve()
    else:
        # Place output directory under the project root (/app), alongside "en" and "scripts" as "json".
        out_dir = script_root / "json"

    for split in args.splits:
        entries = collect_entries_for_split(project_root, split)

        out_json = out_dir / f"sick_{split}.json"
        write_json(entries, out_json)

        out_md = out_dir / f"sick_{split}.md"
        write_markdown(entries, out_md)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())