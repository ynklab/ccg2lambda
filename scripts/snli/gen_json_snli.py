#!/usr/bin/env python3

import argparse
import json
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm


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
    # (this is how ccg2lambda encodes formulas in {dsname}_*.txt.sem.xml).
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


def get_nli_label(zipf: zipfile.ZipFile, pair_id: str) -> str:
    """Look up the NLI gold label for a pair using the corresponding .answer file.

    The gold label is stored in the zip file at:

        results/{pair_id}.answer

    where pair_id is like "{dsname}_train_1.txt", e.g.:

        snli_test_6.txt.sem.xml  ->  snli_test_6.txt  ->  results/snli_test_6.txt.answer

    This function extracts the .answer file from the zip on demand,
    reads the first non-empty line, and returns the label.
    """
    # pair_id is usually like "{dsname}_train_1.txt"; construct the answer file path
    answer_path_in_zip = f"results/{pair_id}.answer"

    try:
        with zipf.open(answer_path_in_zip, "r") as f:
            for line in f:
                label = line.decode("utf-8").strip()
                if not label:
                    continue
                return label
    except Exception:
        return "None"

    return "None"


def xml_to_entry(xml_content: bytes, filename: str, zipf: zipfile.ZipFile) -> Dict[str, str]:
    """
    Convert one XML file (containing a premise/hypothesis pair) into
    a single JSON-ready dict with the required keys.
    
    Args:
        xml_content: XML content as bytes
        filename: Name of the XML file in the zip (e.g., "parsed/snli_train_1.txt.sem.xml")
        zipf: ZipFile object for extracting answer files on demand
    """
    root = ET.fromstring(xml_content)

    # Derive an ID from the file name, e.g.:
    #   parsed/snli_train_1234.txt.sem.xml -> snli_train_1234.txt
    # Remove "parsed/" prefix if present, then strip .sem.xml
    fname = filename
    if "/" in fname:
        fname = fname.split("/")[-1]  # Get just the filename part
    
    # Prefer to strip the trailing '.sem.xml' and keep the '.txt' part,
    # e.g. 'snli_train_1.txt.sem.xml' -> 'snli_train_1.txt'.
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
            f"Expected at least 2 <sentence> elements in {filename}, found {len(sentence_elems)}"
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
        "NLI-gold-label": get_nli_label(zipf, pair_id),
        "premise-logical-formulas": premise_formula,
        "hypothesis-logical-formulas": hypothesis_formula,
    }
    return entry


def collect_entries_for_split(zipf: zipfile.ZipFile, split: str, dsname: str) -> List[Dict[str, str]]:
    """
    Collect all XML files for a given split ('train', 'test', or 'validation')
    and convert them into JSON entries from a zip file.
    Extracts files on demand and removes them after processing to avoid trash.
    
    Args:
        zipf: Already-opened ZipFile object for reading XML files
        split: Dataset split ('train', 'test', or 'validation')
        dsname: Dataset name (e.g., 'snli')
    """
    entries: List[Dict[str, str]] = []
    pattern = f"parsed/{dsname}_{split}_*.txt.sem.xml"
    
    # Create temporary directory for extraction
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        zip_names = zipf.namelist()
        # Look for XML files in parsed/ directory matching the split
        xml_files = sorted([name for name in zip_names 
                           if name.startswith(f"parsed/{dsname}_{split}_") 
                           and name.endswith(".txt.sem.xml")])
        
        if not xml_files:
            print(f"[WARN] No XML files matching {pattern} in zip", file=sys.stderr)
            return entries
        
        # Process files with progress bar
        for filename in tqdm(xml_files, desc=f"{split} split"):
            extracted_path = None
            try:
                # Extract file to temporary directory
                # Create subdirectory structure if needed
                extracted_path = temp_path / filename
                extracted_path.parent.mkdir(parents=True, exist_ok=True)
                zipf.extract(filename, temp_path)
                
                # Read and process the extracted file
                with extracted_path.open("rb") as f:
                    xml_content = f.read()
                entry = xml_to_entry(xml_content, filename, zipf)
                entries.append(entry)
            except Exception as e:
                print(f"[ERROR] Failed to process {filename} from zip: {e}", file=sys.stderr)
            finally:
                # Remove extracted file immediately after processing
                if extracted_path and extracted_path.exists():
                    extracted_path.unlink()
                    # Try to remove parent directory if empty
                    try:
                        extracted_path.parent.rmdir()
                    except OSError:
                        pass  # Directory not empty or doesn't exist

    return entries



def write_json(entries: List[Dict[str, str]], zipf: zipfile.ZipFile, filename: str) -> None:
    """Write a list of dicts to a JSON file in a zip archive (UTF-8, pretty-printed)."""
    json_str = json.dumps(entries, ensure_ascii=False, indent=2)
    zipf.writestr(filename, json_str.encode("utf-8"))
    print(f"[INFO] Wrote {len(entries)} entries to {filename} in zip")


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
def write_markdown(entries: List[Dict[str, str]], zipf: zipfile.ZipFile, filename: str) -> None:
    """Write entries into a Markdown file in a zip archive suitable for manual inspection (e.g., on HackMD)."""
    md_content = []
    md_content.append("# SNLI dataset entries — {}\n\n".format(Path(filename).stem))
    for entry in entries:
        md_content.append("## ID: {}\n\n".format(entry["ID"]))
        md_content.append("**Premise**:\n\n{}\n\n".format(entry["premise"]))
        md_content.append("**Hypothesis**:\n\n{}\n\n".format(entry["hypothesis"]))
        md_content.append("**Gold-label**: {}\n\n".format(entry["NLI-gold-label"]))

        premise_formula = entry.get("premise-logical-formulas", "")
        hypothesis_formula = entry.get("hypothesis-logical-formulas", "")

        if premise_formula:
            md_content.append("**Premise-logical-formulas:**\n\n")
            latex_premise = to_latex_formula(premise_formula)
            md_content.append("$$\n{}\n$$\n\n".format(latex_premise))
        else:
            md_content.append("**Premise-logical-formulas:** –\n\n")

        if hypothesis_formula:
            md_content.append("**Hypothesis-logical-formulas:**\n\n")
            latex_hyp = to_latex_formula(hypothesis_formula)
            md_content.append("$$\n{}\n$$\n\n".format(latex_hyp))
        else:
            md_content.append("**Hypothesis-logical-formulas:** –\n\n")
    
    md_str = "".join(md_content)
    zipf.writestr(filename, md_str.encode("utf-8"))
    print(f"[INFO] Wrote Markdown file to {filename} in zip")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate SNLI JSON files from parsed XML (ccg2lambda)."
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
        default='train,test,validation',
        help='Which splits to process (default: "train,test,validation").',
    )
    parser.add_argument(
        "--dsname",
        type=str,
        default="snli",
        help="Dataset name (default: snli).",
    )
    parser.add_argument(
        "--zip-path",
        type=str,
        default=None,
        help="Path to zip file containing XML files (e.g., snli_result.zip). Default: snli_result.zip in script directory.",
    )
    parser.add_argument(
        "--out-zip-path",
        type=str,
        default=None,
        help="Path to output zip file for JSON/Markdown files (e.g., snli_json.zip). Default: snli_json.zip in script directory.",
    )

    args = parser.parse_args(argv)

    dsname = args.dsname

    # Determine input zip file path
    if args.zip_path is not None:
        zip_path = Path(args.zip_path).resolve()
    else:
        # Default to snli_result.zip in the script directory (same location as ccg_txt_snli.py)
        script_dir = Path(__file__).resolve().parent
        zip_path = script_dir / "snli_result.zip"

    if not zip_path.exists():
        print(f"[ERROR] Zip file not found: {zip_path}", file=sys.stderr)
        return 1

    # Determine output zip file path
    if args.out_zip_path is not None:
        out_zip_path = Path(args.out_zip_path).resolve()
    else:
        # Default to snli_json.zip in the script directory
        script_dir = Path(__file__).resolve().parent
        out_zip_path = script_dir / "snli_json.zip"

    splits = args.splits.split(",")

    # Open both zip files globally to avoid opening every time
    with zipfile.ZipFile(zip_path, "r") as in_zipf, zipfile.ZipFile(out_zip_path, "a", zipfile.ZIP_DEFLATED) as out_zipf:
        for split in splits:
            entries = collect_entries_for_split(in_zipf, split, dsname)

            json_filename = f"{dsname}_{split}.json"
            write_json(entries, out_zipf, json_filename)

            md_filename = f"{dsname}_{split}.md"
            write_markdown(entries, out_zipf, md_filename)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
