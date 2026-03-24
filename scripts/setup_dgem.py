"""
DGEM Data Setup Script
=======================
Downloads and processes all data needed for the Drug-Gene Expression Matching module:
  1. DeepCE pre-computed drug expression profiles (754MB from Google Drive)
  2. CREEDS disease expression signatures (16.8MB JSON)
  3. Drug name mapping (from TxGNN node.csv DrugBank IDs)
  4. Disease name mapping (fuzzy match TxGNN -> CREEDS)
  5. L1000 landmark gene list extraction

Run once:  python scripts/setup_dgem.py
"""

import os
import sys
import json
import pickle
import zipfile
import requests
import csv
import numpy as np
from difflib import SequenceMatcher

# Project root (one level up from scripts/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")
DGEM_FOLDER = os.path.join(DATA_FOLDER, "dgem")
DEEPCE_RAW = os.path.join(DGEM_FOLDER, "deepce_raw")

CREEDS_URL = "https://maayanlab.cloud/CREEDS/download/disease_signatures-v1.0.json"

# Google Drive file ID for DeepCE drugbank_gene_expression.zip
DEEPCE_GDRIVE_ID = "1B2s2HOcB3sWGi3qYcYTcdD4EbLPSnpQN"


def ensure_dirs():
    """Create necessary directories."""
    for d in [DGEM_FOLDER, DEEPCE_RAW]:
        os.makedirs(d, exist_ok=True)
    print(f"[SETUP] Data directory: {DGEM_FOLDER}")


# ─────────────────────────────────────────────────────────────
# Step 1: Download DeepCE pre-computed drug expression profiles
# ─────────────────────────────────────────────────────────────

def download_from_gdrive(file_id, dest_path):
    """Download a large file from Google Drive handling the virus scan warning."""
    # Try multiple approaches since Google Drive changes their download flow

    # Approach 1: Try gdown (best for large files)
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        print("[DOWNLOAD] Using gdown...")
        gdown.download(url, dest_path, quiet=False)
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1_000_000:
            return True
    except ImportError:
        print("[DOWNLOAD] gdown not installed, trying requests approach...")
    except Exception as e:
        print(f"[DOWNLOAD] gdown failed: {e}")

    # Approach 2: Direct download with confirm=t
    session = requests.Session()
    methods = [
        # Method A: Direct export with confirm=t
        {"url": "https://drive.google.com/uc?export=download",
         "params": {"id": file_id, "confirm": "t"}},
        # Method B: Open URL format
        {"url": f"https://drive.usercontent.google.com/download",
         "params": {"id": file_id, "export": "download", "confirm": "t"}},
        # Method C: Classic cookie-based approach
        {"url": "https://drive.google.com/uc?export=download",
         "params": {"id": file_id}},
    ]

    for i, method in enumerate(methods):
        print(f"[DOWNLOAD] Trying method {i+1}/{len(methods)}...")
        try:
            response = session.get(method["url"], params=method["params"],
                                   stream=True, timeout=30)

            # If we got HTML (confirmation page), look for cookie/token
            content_type = response.headers.get("Content-Type", "")
            if content_type.startswith("text/html"):
                # Try to extract confirmation token from cookies
                token = None
                for key, value in response.cookies.items():
                    if key.startswith("download_warning"):
                        token = value
                        break
                if token:
                    response = session.get(
                        method["url"],
                        params={**method["params"], "confirm": token},
                        stream=True, timeout=30
                    )
                    content_type = response.headers.get("Content-Type", "")

                # If still HTML, skip to next method
                if content_type.startswith("text/html"):
                    continue

            # Download the file
            total = int(response.headers.get("content-length", 0))
            downloaded = 0
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192 * 16):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = downloaded / total * 100
                            mb = downloaded / 1e6
                            total_mb = total / 1e6
                            print(f"\r[DOWNLOAD] {mb:.1f} / {total_mb:.1f} MB "
                                  f"({pct:.1f}%)", end="", flush=True)
                        else:
                            print(f"\r[DOWNLOAD] {downloaded / 1e6:.1f} MB",
                                  end="", flush=True)
            print()

            # Verify we got actual data
            if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1_000_000:
                return True
            else:
                # File too small, probably error page
                if os.path.exists(dest_path):
                    with open(dest_path, "r", errors="ignore") as f:
                        head = f.read(500)
                    if "<html" in head.lower():
                        os.remove(dest_path)
                continue

        except Exception as e:
            print(f"\n[DOWNLOAD] Method {i+1} failed: {e}")
            continue

    return False


def download_deepce_profiles():
    """Download and extract DeepCE drug expression profiles."""
    zip_path = os.path.join(DGEM_FOLDER, "drugbank_gene_expression.zip")

    if os.path.exists(os.path.join(DEEPCE_RAW, "_extraction_complete")):
        print("[STEP 1] DeepCE profiles already downloaded and extracted.")
        return True

    if not os.path.exists(zip_path):
        print("[STEP 1] Downloading DeepCE drug expression profiles (~754 MB)...")
        print("[STEP 1] This is a one-time download. Please be patient.")
        success = download_from_gdrive(DEEPCE_GDRIVE_ID, zip_path)
        if not success:
            print("[STEP 1] ERROR: Automated Google Drive download failed.")
            print("[STEP 1] Option A - Install gdown (recommended):")
            print(f"  pip install gdown")
            print(f"  Then re-run this script")
            print("[STEP 1] Option B - Download manually:")
            print(f"  1. Go to: https://drive.google.com/file/d/{DEEPCE_GDRIVE_ID}/view")
            print(f"  2. Click the download button")
            print(f"  3. Save as: {zip_path}")
            print(f"  4. Re-run this script")
            return False
    else:
        print(f"[STEP 1] Zip already exists: {zip_path}")

    # Extract
    print("[STEP 1] Extracting zip file...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DEEPCE_RAW)
        # Mark extraction complete
        with open(os.path.join(DEEPCE_RAW, "_extraction_complete"), "w") as f:
            f.write("done")
        print("[STEP 1] Extraction complete.")
        return True
    except zipfile.BadZipFile:
        print("[STEP 1] ERROR: Downloaded file is not a valid zip.")
        print("[STEP 1] Try deleting it and re-downloading manually.")
        os.remove(zip_path)
        return False


# ─────────────────────────────────────────────────────────────
# Step 2: Process DeepCE profiles into HDF5
# ─────────────────────────────────────────────────────────────

def process_deepce_profiles():
    """
    Process DeepCE CSV files into drug profiles keyed by DrugBank ID.

    DeepCE data format:
      - drug_id.csv: single row, 11,179 columns = BRD IDs (drug identifiers)
      - a375.csv, a549.csv, ...: 11,179 rows (drugs) x 978 columns (genes)
        NO header row — columns are L1000 gene expression values
      - Row i in each cell-line CSV corresponds to BRD ID at position i in drug_id.csv

    We need to:
      1. Read BRD IDs from drug_id.csv
      2. Read expression matrices from cell-line CSVs
      3. Map BRD IDs -> DrugBank IDs via SMILES matching or name matching
      4. Average across cell lines, save keyed by DrugBank ID
    """
    import pandas as pd

    output_path = os.path.join(DGEM_FOLDER, "drug_profiles.pkl")
    genes_path = os.path.join(DGEM_FOLDER, "l1000_genes.json")

    if os.path.exists(output_path) and os.path.exists(genes_path):
        print("[STEP 2] Drug profiles already processed.")
        return True

    deepce_dir = os.path.join(DEEPCE_RAW, "drugbank_gene_expression")
    drug_id_file = os.path.join(deepce_dir, "drug_id.csv")

    if not os.path.exists(drug_id_file):
        print(f"[STEP 2] ERROR: drug_id.csv not found at {drug_id_file}")
        return False

    # Step 2a: Read BRD IDs
    print("[STEP 2a] Reading BRD drug IDs...")
    drug_id_df = pd.read_csv(drug_id_file)
    brd_ids = list(drug_id_df.columns)
    n_drugs = len(brd_ids)
    print(f"[STEP 2a] Found {n_drugs} BRD IDs")

    # Step 2b: Read and average cell-line expression matrices
    cell_lines = ["a375", "a549", "ha1e", "hela", "ht29", "mcf7", "pc3", "yapc"]
    expression_sum = None
    n_celllines_loaded = 0

    for cl in cell_lines:
        cl_file = os.path.join(deepce_dir, f"{cl}.csv")
        if not os.path.exists(cl_file):
            print(f"[STEP 2b] WARNING: {cl}.csv not found, skipping.")
            continue

        print(f"[STEP 2b] Reading {cl}.csv...", end=" ", flush=True)
        # No header row; 11,179 rows x 978 columns
        df = pd.read_csv(cl_file, header=None)
        print(f"shape={df.shape}")

        if df.shape[0] != n_drugs:
            print(f"[STEP 2b] WARNING: {cl}.csv has {df.shape[0]} rows "
                  f"but expected {n_drugs}. Skipping.")
            continue

        mat = df.values.astype(np.float32)
        if expression_sum is None:
            expression_sum = mat.copy()
        else:
            expression_sum += mat
        n_celllines_loaded += 1

    if n_celllines_loaded == 0:
        print("[STEP 2b] ERROR: No cell-line files loaded.")
        return False

    # Average across cell lines
    expression_avg = expression_sum / n_celllines_loaded
    n_genes = expression_avg.shape[1]
    print(f"[STEP 2b] Averaged {n_celllines_loaded} cell lines, "
          f"{n_drugs} drugs x {n_genes} genes")

    # Step 2c: Save L1000 gene list (we don't have gene names from DeepCE
    # output, so generate positional indices; the CREEDS signatures will
    # need to be matched by gene name separately)
    # Actually, get the gene names from the DeepCE training data if available
    gene_names = _get_l1000_gene_names(n_genes)
    with open(genes_path, "w") as f:
        json.dump(gene_names, f, indent=2)
    print(f"[STEP 2c] Saved {len(gene_names)} gene names to {genes_path}")

    # Step 2d: Build BRD -> DrugBank mapping
    print("[STEP 2d] Building BRD -> DrugBank ID mapping...")
    brd_to_drugbank = _build_brd_to_drugbank_mapping(brd_ids)
    mapped_count = sum(1 for v in brd_to_drugbank.values() if v is not None)
    print(f"[STEP 2d] Mapped {mapped_count}/{n_drugs} BRD IDs to DrugBank IDs")

    # Step 2e: Save profiles keyed by DrugBank ID
    drug_profiles = {}
    unmapped = 0
    for i, brd_id in enumerate(brd_ids):
        db_id = brd_to_drugbank.get(brd_id)
        if db_id:
            drug_profiles[db_id] = expression_avg[i]
        else:
            # Store under BRD ID as fallback
            drug_profiles[brd_id] = expression_avg[i]
            unmapped += 1

    with open(output_path, "wb") as f:
        pickle.dump(drug_profiles, f, protocol=4)

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"[STEP 2e] Saved {len(drug_profiles)} drug profiles "
          f"({mapped_count} with DrugBank IDs, {unmapped} BRD-only) "
          f"to {output_path} ({size_mb:.1f} MB)")
    return True


# L1000 landmark gene list (978 genes) - canonical order from LINCS
# Source: https://clue.io/command?q=/gene-space
_L1000_GENES_URL = ("https://s3.amazonaws.com/macchiato.clue.io/builds/"
                    "LINCS2020/geneinfo_beta.txt")


def _get_l1000_gene_names(expected_n):
    """Get L1000 landmark gene names."""
    # Try to download from LINCS
    try:
        print("[GENES] Downloading L1000 gene info from LINCS...")
        resp = requests.get(_L1000_GENES_URL, timeout=30)
        resp.raise_for_status()

        import io
        import pandas as pd
        gene_df = pd.read_csv(io.StringIO(resp.text), sep="\t")
        # Filter to landmark genes
        landmarks = gene_df[gene_df["feature_space"] == "landmark"]
        gene_names = landmarks["gene_symbol"].tolist()
        if len(gene_names) >= expected_n:
            print(f"[GENES] Got {len(gene_names)} landmark genes from LINCS")
            return gene_names[:expected_n]
    except Exception as e:
        print(f"[GENES] Could not download gene info: {e}")

    # Fallback: use DeepCE training data gene names if available
    deepce_sig_files = [
        os.path.join(DEEPCE_RAW, "signature_train.csv"),
        os.path.join(PROJECT_ROOT, "DeepCE", "data", "signature_train.csv"),
    ]
    import pandas as pd
    for sig_file in deepce_sig_files:
        if os.path.exists(sig_file):
            try:
                df = pd.read_csv(sig_file, nrows=1)
                # Gene columns are those that aren't metadata
                meta = {"sig_id", "pert_id", "pert_type", "cell_id",
                        "pert_idose", "pert_itime"}
                genes = [c for c in df.columns if c not in meta]
                if len(genes) >= expected_n:
                    print(f"[GENES] Got {len(genes)} genes from {sig_file}")
                    return genes[:expected_n]
            except Exception:
                pass

    # Last fallback: positional indices
    print(f"[GENES] Using positional indices for {expected_n} genes")
    return [f"gene_{i}" for i in range(expected_n)]


def _build_brd_to_drugbank_mapping(brd_ids):
    """
    Build a mapping from BRD IDs to DrugBank IDs.

    Strategy: Download the Broad Repurposing Hub samples file which maps
    BRD IDs to drug names, then match drug names to TxGNN's DrugBank IDs.
    """
    import pandas as pd

    mapping = {brd: None for brd in brd_ids}
    brd_set = set(brd_ids)

    # Load TxGNN drug mapping (name -> DrugBank ID)
    drug_map_path = os.path.join(DGEM_FOLDER, "drug_id_mapping.json")
    if not os.path.exists(drug_map_path):
        print("[MAPPING] No TxGNN drug mapping found. Cannot build BRD->DrugBank mapping.")
        return mapping

    with open(drug_map_path, "r") as f:
        txgnn_map = json.load(f)

    # Reverse: DrugBank ID -> drug name (lowercase)
    db_to_name = {v: k for k, v in txgnn_map.get("name_to_id", {}).items()}
    # Name (lowercase) -> DrugBank ID
    name_to_db = {k.lower(): v for k, v in txgnn_map.get("name_to_id", {}).items()}

    # Strategy 1: Try Broad Repurposing Hub samples file
    repurposing_url = ("https://s3.amazonaws.com/data.clue.io/repurposing/"
                       "downloads/repurposing_samples_20200324.txt")
    try:
        print("[MAPPING] Downloading Broad Repurposing Hub samples...")
        resp = requests.get(repurposing_url, timeout=60)
        resp.raise_for_status()

        import io
        # Skip comment lines (start with !)
        lines = [l for l in resp.text.split("\n") if not l.startswith("!")]
        hub_df = pd.read_csv(io.StringIO("\n".join(lines)), sep="\t")

        # Map via drug name
        if "pert_iname" in hub_df.columns and "broad_id" in hub_df.columns:
            for _, row in hub_df.iterrows():
                brd = str(row.get("broad_id", "")).strip()
                name = str(row.get("pert_iname", "")).strip().lower()
                if brd in brd_set and name in name_to_db:
                    mapping[brd] = name_to_db[name]

        matched = sum(1 for v in mapping.values() if v is not None)
        print(f"[MAPPING] Repurposing Hub: matched {matched} BRD IDs to DrugBank")

    except Exception as e:
        print(f"[MAPPING] Repurposing Hub download failed: {e}")

    # Strategy 2: Try LINCS pert_info from GEO (larger, ~20K compounds)
    try:
        pert_info_url = ("https://s3.amazonaws.com/macchiato.clue.io/builds/"
                         "LINCS2020/compoundinfo_beta.txt")
        print("[MAPPING] Downloading LINCS compound info...")
        resp = requests.get(pert_info_url, timeout=60)
        resp.raise_for_status()

        import io
        pert_df = pd.read_csv(io.StringIO(resp.text), sep="\t")

        name_col = None
        id_col = None
        for c in ["cmap_name", "pert_iname"]:
            if c in pert_df.columns:
                name_col = c
                break
        for c in ["pert_id", "broad_id"]:
            if c in pert_df.columns:
                id_col = c
                break

        if name_col and id_col:
            for _, row in pert_df.iterrows():
                brd = str(row[id_col]).strip()
                name = str(row[name_col]).strip().lower()
                if brd in brd_set and mapping.get(brd) is None and name in name_to_db:
                    mapping[brd] = name_to_db[name]

        matched = sum(1 for v in mapping.values() if v is not None)
        print(f"[MAPPING] LINCS compound info: total matched = {matched}")

    except Exception as e:
        print(f"[MAPPING] LINCS compound info download failed: {e}")

    return mapping


# ─────────────────────────────────────────────────────────────
# Step 3: Download CREEDS disease signatures
# ─────────────────────────────────────────────────────────────

def download_creeds():
    """Download CREEDS disease expression signatures."""
    raw_path = os.path.join(DGEM_FOLDER, "creeds_raw.json")

    if os.path.exists(raw_path):
        print("[STEP 3] CREEDS data already downloaded.")
        return True

    print("[STEP 3] Downloading CREEDS disease signatures (16.8 MB)...")
    try:
        response = requests.get(CREEDS_URL, timeout=120)
        response.raise_for_status()
        with open(raw_path, "wb") as f:
            f.write(response.content)
        print(f"[STEP 3] Saved to {raw_path}")
        return True
    except Exception as e:
        print(f"[STEP 3] ERROR: Failed to download CREEDS: {e}")
        print(f"[STEP 3] Try manually: download from {CREEDS_URL}")
        print(f"[STEP 3] Save as: {raw_path}")
        return False


def process_creeds_signatures():
    """Process CREEDS JSON into disease signature vectors aligned to L1000 genes."""
    raw_path = os.path.join(DGEM_FOLDER, "creeds_raw.json")
    output_path = os.path.join(DGEM_FOLDER, "disease_signatures.pkl")
    genes_path = os.path.join(DGEM_FOLDER, "l1000_genes.json")

    if os.path.exists(output_path):
        print("[STEP 3b] Disease signatures already processed.")
        return True

    if not os.path.exists(raw_path):
        print("[STEP 3b] ERROR: CREEDS raw data not found.")
        return False

    # Load L1000 gene list
    if os.path.exists(genes_path):
        with open(genes_path, "r") as f:
            l1000_genes = json.load(f)
    else:
        print("[STEP 3b] WARNING: L1000 gene list not found. Will store full signatures.")
        l1000_genes = None

    print("[STEP 3b] Processing CREEDS disease signatures...")
    with open(raw_path, "r", encoding="utf-8") as f:
        creeds_data = json.load(f)

    # Filter to human only
    human_sigs = [s for s in creeds_data if s.get("organism", "").lower() == "human"]
    print(f"[STEP 3b] Total signatures: {len(creeds_data)}, Human: {len(human_sigs)}")

    # Build gene set from L1000 for fast lookup
    l1000_set = set(l1000_genes) if l1000_genes else None

    # Process each signature into a vector
    disease_signatures = {}  # disease_name -> list of {vector, geo_id, cell_type}
    gene_to_idx = {g: i for i, g in enumerate(l1000_genes)} if l1000_genes else None

    for sig in human_sigs:
        disease_name = sig.get("disease_name", "").strip().lower()
        if not disease_name:
            continue

        # Build expression vector from up/down genes
        # CREEDS format: up_genes = [[gene_symbol, cd_coefficient], ...]
        up_genes = sig.get("up_genes", [])
        down_genes = sig.get("down_genes", [])

        if l1000_genes and gene_to_idx:
            # Create vector aligned to L1000 genes
            vector = np.zeros(len(l1000_genes), dtype=np.float32)
            matched = 0
            for gene_name, cd_value in up_genes:
                if gene_name in gene_to_idx:
                    vector[gene_to_idx[gene_name]] = float(cd_value)
                    matched += 1
            for gene_name, cd_value in down_genes:
                if gene_name in gene_to_idx:
                    vector[gene_to_idx[gene_name]] = float(cd_value)
                    matched += 1
        else:
            # Store as dict if no L1000 gene list
            vector = {}
            matched = 0
            for gene_name, cd_value in up_genes:
                vector[gene_name] = float(cd_value)
                matched += 1
            for gene_name, cd_value in down_genes:
                vector[gene_name] = float(cd_value)
                matched += 1

        if disease_name not in disease_signatures:
            disease_signatures[disease_name] = []

        disease_signatures[disease_name].append({
            "vector": vector,
            "geo_id": sig.get("geo_id", ""),
            "cell_type": sig.get("cell_type", ""),
            "do_id": sig.get("do_id", ""),
            "l1000_matched": matched,
        })

    # Average multiple signatures per disease
    disease_avg = {}
    for disease_name, sigs in disease_signatures.items():
        if l1000_genes:
            vectors = [s["vector"] for s in sigs]
            disease_avg[disease_name] = {
                "vector": np.mean(vectors, axis=0).astype(np.float32),
                "n_signatures": len(sigs),
                "geo_ids": [s["geo_id"] for s in sigs],
                "avg_l1000_matched": np.mean([s["l1000_matched"] for s in sigs]),
            }
        else:
            disease_avg[disease_name] = {
                "signatures": sigs,
                "n_signatures": len(sigs),
            }

    with open(output_path, "wb") as f:
        pickle.dump(disease_avg, f, protocol=4)

    print(f"[STEP 3b] Saved {len(disease_avg)} unique disease signatures to {output_path}")

    # Report top diseases
    top = sorted(disease_avg.items(), key=lambda x: x[1]["n_signatures"], reverse=True)[:10]
    print("[STEP 3b] Top diseases by signature count:")
    for name, info in top:
        print(f"    {name}: {info['n_signatures']} signatures")

    return True


# ─────────────────────────────────────────────────────────────
# Step 4: Build drug name mapping (TxGNN -> DrugBank ID)
# ─────────────────────────────────────────────────────────────

def build_drug_mapping():
    """Extract DrugBank IDs from TxGNN's node.csv."""
    output_path = os.path.join(DGEM_FOLDER, "drug_id_mapping.json")

    if os.path.exists(output_path):
        print("[STEP 4] Drug ID mapping already built.")
        return True

    nodes_file = os.path.join(DATA_FOLDER, "node.csv")
    if not os.path.exists(nodes_file):
        print(f"[STEP 4] ERROR: node.csv not found at {nodes_file}")
        print("[STEP 4] Run TxGNN data loading first (python main.py --discover)")
        return False

    print("[STEP 4] Building drug name -> DrugBank ID mapping from node.csv...")
    import pandas as pd
    df = pd.read_csv(nodes_file, sep="\t", on_bad_lines="skip")

    # Strip quotes
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip('"')

    drugs = df[df["node_type"] == "drug"].sort_values("node_index")

    # Build mappings
    name_to_id = {}    # drug_name -> DrugBank ID
    idx_to_id = {}     # per_type_idx -> DrugBank ID
    id_to_name = {}    # DrugBank ID -> drug_name

    for per_type_idx, (_, row) in enumerate(drugs.iterrows()):
        drug_name = row["node_name"]
        drugbank_id = row["node_id"]
        name_to_id[drug_name] = drugbank_id
        idx_to_id[str(per_type_idx)] = drugbank_id
        id_to_name[drugbank_id] = drug_name

    mapping = {
        "name_to_id": name_to_id,
        "idx_to_id": idx_to_id,
        "id_to_name": id_to_name,
    }

    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"[STEP 4] Mapped {len(name_to_id)} drug names to DrugBank IDs")
    print(f"[STEP 4] Sample: {list(name_to_id.items())[:5]}")
    return True


# ─────────────────────────────────────────────────────────────
# Step 5: Build disease name mapping (TxGNN -> CREEDS)
# ─────────────────────────────────────────────────────────────

def build_disease_mapping():
    """Fuzzy-match TxGNN disease names to CREEDS disease names."""
    output_path = os.path.join(DGEM_FOLDER, "disease_name_mapping.json")
    sigs_path = os.path.join(DGEM_FOLDER, "disease_signatures.pkl")

    if os.path.exists(output_path):
        print("[STEP 5] Disease name mapping already built.")
        return True

    nodes_file = os.path.join(DATA_FOLDER, "node.csv")
    if not os.path.exists(nodes_file) or not os.path.exists(sigs_path):
        print("[STEP 5] ERROR: Required files missing (node.csv or disease_signatures.pkl)")
        return False

    print("[STEP 5] Building TxGNN -> CREEDS disease name mapping...")

    # Load TxGNN disease names
    import pandas as pd
    df = pd.read_csv(nodes_file, sep="\t", on_bad_lines="skip")
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip('"')

    diseases = df[df["node_type"] == "disease"].sort_values("node_index")
    txgnn_diseases = {}
    for per_type_idx, (_, row) in enumerate(diseases.iterrows()):
        txgnn_diseases[per_type_idx] = row["node_name"].lower().strip()

    # Load CREEDS disease names
    with open(sigs_path, "rb") as f:
        disease_sigs = pickle.load(f)
    creeds_names = list(disease_sigs.keys())

    # Fuzzy match
    mapping = {}  # txgnn_name -> creeds_name
    match_count = 0

    creeds_set = set(creeds_names)

    for idx, txgnn_name in txgnn_diseases.items():
        # 1. Exact match
        if txgnn_name in creeds_set:
            mapping[txgnn_name] = txgnn_name
            match_count += 1
            continue

        # 2. Fuzzy match with stricter threshold
        # Use higher threshold for short names to avoid false matches
        # (e.g., 'brown syndrome' matching 'down syndrome')
        best_score = 0
        best_match = None
        for creeds_name in creeds_names:
            score = SequenceMatcher(None, txgnn_name, creeds_name).ratio()
            if score > best_score:
                best_score = score
                best_match = creeds_name

        # Require higher similarity for short names (< 20 chars)
        threshold = 0.90 if len(txgnn_name) < 20 else 0.85
        if best_score >= threshold:
            mapping[txgnn_name] = best_match
            match_count += 1

    with open(output_path, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"[STEP 5] Matched {match_count} of {len(txgnn_diseases)} TxGNN diseases "
          f"to CREEDS ({match_count/len(txgnn_diseases)*100:.1f}%)")

    # Show sample matches
    sample = list(mapping.items())[:10]
    print("[STEP 5] Sample matches:")
    for txgnn, creeds in sample:
        print(f"    '{txgnn}' -> '{creeds}'")

    return True


# ─────────────────────────────────────────────────────────────
# Step 6: Coverage report
# ─────────────────────────────────────────────────────────────

def coverage_report():
    """Print coverage statistics."""
    print("\n" + "=" * 60)
    print("  DGEM DATA COVERAGE REPORT")
    print("=" * 60)

    # Drug profiles
    profiles_path = os.path.join(DGEM_FOLDER, "drug_profiles.pkl")
    if os.path.exists(profiles_path):
        with open(profiles_path, "rb") as f:
            profiles = pickle.load(f)
        print(f"\n  Drug expression profiles: {len(profiles)} drugs")
    else:
        print("\n  Drug expression profiles: NOT AVAILABLE")

    # Drug mapping
    drug_map_path = os.path.join(DGEM_FOLDER, "drug_id_mapping.json")
    if os.path.exists(drug_map_path):
        with open(drug_map_path, "r") as f:
            drug_map = json.load(f)
        txgnn_drugs = set(drug_map["name_to_id"].values())  # DrugBank IDs
        print(f"  TxGNN drugs with DrugBank IDs: {len(txgnn_drugs)}")

        # Check overlap with DeepCE
        if os.path.exists(profiles_path):
            deepce_ids = set(profiles.keys())
            overlap = txgnn_drugs.intersection(deepce_ids)
            print(f"  TxGNN-DeepCE drug overlap: {len(overlap)} "
                  f"({len(overlap)/len(txgnn_drugs)*100:.1f}% of TxGNN drugs)")
    else:
        print("  Drug ID mapping: NOT AVAILABLE")

    # Disease signatures
    sigs_path = os.path.join(DGEM_FOLDER, "disease_signatures.pkl")
    if os.path.exists(sigs_path):
        with open(sigs_path, "rb") as f:
            sigs = pickle.load(f)
        print(f"\n  CREEDS disease signatures: {len(sigs)} unique diseases")
    else:
        print("\n  CREEDS disease signatures: NOT AVAILABLE")

    # Disease mapping
    disease_map_path = os.path.join(DGEM_FOLDER, "disease_name_mapping.json")
    if os.path.exists(disease_map_path):
        with open(disease_map_path, "r") as f:
            disease_map = json.load(f)
        print(f"  TxGNN diseases matched to CREEDS: {len(disease_map)}")
    else:
        print("  Disease name mapping: NOT AVAILABLE")

    # L1000 genes
    genes_path = os.path.join(DGEM_FOLDER, "l1000_genes.json")
    if os.path.exists(genes_path):
        with open(genes_path, "r") as f:
            genes = json.load(f)
        print(f"\n  L1000 gene columns: {len(genes)}")
    else:
        print("\n  L1000 gene list: NOT AVAILABLE")

    # Check specific drugs/diseases of interest
    print("\n  --- Specific Checks ---")

    if os.path.exists(drug_map_path) and os.path.exists(profiles_path):
        for drug in ["Sulindac", "Metformin", "Aspirin"]:
            db_id = drug_map.get("name_to_id", {}).get(drug)
            has_profile = db_id in profiles if db_id else False
            print(f"  {drug}: DrugBank={db_id}, Has DeepCE profile={has_profile}")

    if os.path.exists(disease_map_path):
        for disease in ["fragile x syndrome", "alzheimer disease", "type 2 diabetes mellitus"]:
            mapped = disease_map.get(disease, "NOT MAPPED")
            print(f"  '{disease}': CREEDS match='{mapped}'")

    print("\n" + "=" * 60)


# ─────────────────────────────────────────────────────────────
# Step 7: Enrich sparse disease signatures by reprocessing GEO
# ─────────────────────────────────────────────────────────────


def enrich_sparse_signatures(sparsity_threshold=0.90):
    """
    Enrich disease signatures that are too sparse by reprocessing their
    source GEO microarray data through the GPL platform annotation.

    CREEDS signatures are built from pre-computed DEGs (up/down gene lists),
    but most DEGs don't overlap with L1000 landmark genes. By going back to
    the raw GEO microarray data (which has expression values for ALL genes
    including L1000 genes), we can build much denser L1000-aligned signatures.

    Requires: pip install GEOparse

    Args:
        sparsity_threshold: Signatures with more zeros than this are enriched.
    """
    sigs_path = os.path.join(DGEM_FOLDER, "disease_signatures.pkl")
    genes_path = os.path.join(DGEM_FOLDER, "l1000_genes.json")
    raw_path = os.path.join(DGEM_FOLDER, "creeds_raw.json")

    if not os.path.exists(sigs_path) or not os.path.exists(genes_path):
        print("[STEP 7] Skipping enrichment: signatures or gene list not found.")
        return False

    try:
        import GEOparse
        import pandas as pd
    except ImportError:
        print("[STEP 7] GEOparse not installed. Install with: pip install GEOparse")
        print("[STEP 7] Skipping GEO enrichment.")
        return True

    with open(sigs_path, "rb") as f:
        disease_sigs = pickle.load(f)
    with open(genes_path, "r") as f:
        l1000_genes = json.load(f)

    l1000_set = set(l1000_genes)
    gene_to_idx = {g: i for i, g in enumerate(l1000_genes)}

    # Identify sparse signatures
    sparse_diseases = []
    for disease_name, sig_data in disease_sigs.items():
        vector = sig_data.get("vector")
        if isinstance(vector, np.ndarray):
            nonzero = np.count_nonzero(vector)
            sparsity = 1.0 - (nonzero / len(vector))
            if sparsity > sparsity_threshold:
                sparse_diseases.append((disease_name, nonzero, sparsity))

    if not sparse_diseases:
        print("[STEP 7] No sparse disease signatures found.")
        return True

    print(f"[STEP 7] Found {len(sparse_diseases)} sparse disease signatures "
          f"(>{sparsity_threshold*100:.0f}% zeros)")

    # Load CREEDS raw data to get GEO IDs for each disease
    if not os.path.exists(raw_path):
        print("[STEP 7] CREEDS raw data not found. Cannot enrich.")
        return False

    with open(raw_path, "r", encoding="utf-8") as f:
        creeds_data = json.load(f)

    # Map disease names to their GEO datasets
    disease_to_geo = {}
    for sig in creeds_data:
        if sig.get("organism", "").lower() != "human":
            continue
        dname = sig.get("disease_name", "").strip().lower()
        geo_id = sig.get("geo_id", "")
        if dname and geo_id:
            if dname not in disease_to_geo:
                disease_to_geo[dname] = set()
            disease_to_geo[dname].add(geo_id)

    enriched_count = 0
    # Process each sparse disease
    for disease_name, old_nonzero, sparsity in sparse_diseases:
        geo_ids = disease_to_geo.get(disease_name, set())
        if not geo_ids:
            continue

        # Get all CREEDS entries for this disease to identify sample groups
        disease_entries = [
            s for s in creeds_data
            if s.get("disease_name", "").strip().lower() == disease_name
            and s.get("organism", "").lower() == "human"
        ]

        new_nonzero = _reprocess_geo_for_disease(
            disease_name, geo_ids, disease_entries, disease_sigs,
            l1000_genes, l1000_set, gene_to_idx
        )

        if new_nonzero is not None and new_nonzero > old_nonzero:
            enriched_count += 1
            print(f"[STEP 7] '{disease_name}': {old_nonzero} -> {new_nonzero} "
                  f"non-zero L1000 genes")

    if enriched_count > 0:
        with open(sigs_path, "wb") as f:
            pickle.dump(disease_sigs, f, protocol=4)
        print(f"\n[STEP 7] Enriched {enriched_count} disease signatures. Saved.")
    else:
        print("\n[STEP 7] No signatures could be enriched from GEO source data.")

    return True


def _reprocess_geo_for_disease(disease_name, geo_ids, creeds_entries,
                                disease_sigs, l1000_genes, l1000_set, gene_to_idx):
    """
    Reprocess GEO microarray data for a disease to build a denser L1000 signature.

    Instead of using only CREEDS pre-computed DEGs (which may not overlap L1000),
    this downloads the full GEO dataset, maps probes to genes via the GPL platform
    annotation, and computes fold changes for ALL L1000 genes.

    Returns:
        New non-zero gene count, or None on failure.
    """
    import GEOparse
    import pandas as pd

    for geo_id in geo_ids:
        try:
            print(f"[STEP 7] Fetching {geo_id} for '{disease_name}'...")
            gse = GEOparse.get_GEO(geo=geo_id, destdir=DEEPCE_RAW, silent=True)

            if not gse.gpls:
                print(f"    {geo_id}: No platform annotation (RNA-seq?). Skipping.")
                continue

            # Get platform annotation for probe -> gene symbol mapping
            gpl_name = list(gse.gpls.keys())[0]
            gpl = gse.gpls[gpl_name]

            # Find the gene symbol column in GPL
            gene_col = None
            for col in gpl.table.columns:
                if col.upper() in ("GENE_SYMBOL", "GENE SYMBOL", "SYMBOL",
                                   "Gene Symbol", "Gene_Symbol"):
                    gene_col = col
                    break
            if gene_col is None:
                for col in gpl.table.columns:
                    if "gene_symbol" in col.lower() or "symbol" in col.lower():
                        gene_col = col
                        break

            if gene_col is None:
                print(f"    {geo_id}: No gene symbol column in {gpl_name}. Skipping.")
                continue

            # Build probe ID -> gene symbol mapping
            # Probe IDs can be numeric (Agilent) or strings (Affymetrix)
            # Gene symbols may contain multiple genes (e.g. "DDR1 /// MIR4640")
            probe_to_gene = {}
            for _, row in gpl.table.iterrows():
                gs = row.get(gene_col)
                if pd.notna(gs) and str(gs).strip():
                    # Take first gene if multiple (separated by ///)
                    symbol = str(gs).split("///")[0].strip()
                    if symbol:
                        probe_to_gene[str(row["ID"]).strip()] = symbol

            l1000_probes = sum(1 for g in probe_to_gene.values() if g in l1000_set)
            print(f"    {geo_id}: {len(probe_to_gene)} probes with genes, "
                  f"{l1000_probes} map to L1000")

            if l1000_probes < 50:
                print(f"    {geo_id}: Too few L1000 probes. Skipping.")
                continue

            # Classify samples into disease vs control from titles
            disease_samples = []
            control_samples = []
            disease_keywords = _get_disease_keywords(disease_name)

            for gsm_name, gsm in gse.gsms.items():
                title = gsm.metadata.get("title", [""])[0].lower()
                source = gsm.metadata.get("source_name_ch1", [""])[0].lower()
                chars = " ".join(gsm.metadata.get("characteristics_ch1", [])).lower()
                combined = f"{title} {source} {chars}"

                is_disease = any(kw in combined for kw in disease_keywords)
                is_control = any(kw in combined for kw in
                               ["control", "normal", "healthy", "unaffected",
                                "wild type", "wild-type", "wildtype", "wt "])

                if is_disease and not is_control:
                    disease_samples.append(gsm_name)
                elif is_control and not is_disease:
                    control_samples.append(gsm_name)

            if not disease_samples or not control_samples:
                print(f"    {geo_id}: Could not identify disease/control groups "
                      f"(disease={len(disease_samples)}, control={len(control_samples)})")
                continue

            print(f"    {geo_id}: {len(disease_samples)} disease, "
                  f"{len(control_samples)} control samples")

            # Collect L1000 gene expression values per group
            disease_vals = {g: [] for g in l1000_genes}
            control_vals = {g: [] for g in l1000_genes}

            for group_samples, group_vals in [
                (disease_samples, disease_vals),
                (control_samples, control_vals)
            ]:
                for gsm_name in group_samples:
                    table = gse.gsms[gsm_name].table
                    if table is None or len(table) == 0:
                        continue
                    for _, row in table.iterrows():
                        pid = str(row["ID_REF"]).strip()
                        # Handle numeric IDs stored as floats (e.g. 3.0 -> "3")
                        if pid.endswith(".0"):
                            pid = pid[:-2]
                        if pid in probe_to_gene:
                            gene = probe_to_gene[pid]
                            if gene in l1000_set and pd.notna(row.get("VALUE")):
                                try:
                                    group_vals[gene].append(float(row["VALUE"]))
                                except (ValueError, TypeError):
                                    pass

            # Compute fold changes for L1000 genes
            new_vector = np.zeros(len(l1000_genes), dtype=np.float32)
            genes_filled = 0
            for gene in l1000_genes:
                if disease_vals[gene] and control_vals[gene]:
                    d_mean = np.mean(disease_vals[gene])
                    c_mean = np.mean(control_vals[gene])
                    fc = d_mean - c_mean  # log-scale values
                    new_vector[gene_to_idx[gene]] = fc
                    genes_filled += 1

            new_nonzero = np.count_nonzero(new_vector)
            old_nonzero = np.count_nonzero(disease_sigs[disease_name]["vector"])

            if new_nonzero > old_nonzero:
                disease_sigs[disease_name]["vector"] = new_vector
                disease_sigs[disease_name]["avg_l1000_matched"] = genes_filled
                return new_nonzero
            else:
                print(f"    {geo_id}: Reprocessed ({new_nonzero} genes) not better "
                      f"than existing ({old_nonzero})")

        except Exception as e:
            print(f"    {geo_id}: Error: {e}")

    return None


def _get_disease_keywords(disease_name):
    """Get keywords to identify disease samples in GEO metadata."""
    keywords = [disease_name]

    # Add common abbreviations/aliases
    aliases = {
        "fragile x syndrome": ["fmr1", "fragile x", "fxs", "fmr1-fm"],
        "alzheimer's disease": ["alzheimer", "ad brain"],
        "parkinson's disease": ["parkinson", "pd brain"],
        "breast cancer": ["breast tumor", "breast carcinoma"],
        "type 2 diabetes mellitus": ["t2d", "type 2 diabetes", "diabetic"],
    }

    extra = aliases.get(disease_name, [])
    keywords.extend(extra)

    # Also add words from the disease name
    words = disease_name.split()
    if len(words) >= 2:
        keywords.append(" ".join(words[:2]))

    return keywords


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  DGEM DATA SETUP")
    print("  Drug-Gene Expression Matching Module")
    print("=" * 60)

    ensure_dirs()

    # Step 1: Download DeepCE profiles
    step1_ok = download_deepce_profiles()

    # Step 2: Process DeepCE profiles
    step2_ok = False
    if step1_ok:
        step2_ok = process_deepce_profiles()
    else:
        print("[STEP 2] SKIPPED (DeepCE download failed)")

    # Step 3: Download and process CREEDS
    step3_ok = download_creeds()
    step3b_ok = False
    if step3_ok:
        step3b_ok = process_creeds_signatures()

    # Step 4: Build drug mapping
    step4_ok = build_drug_mapping()

    # Step 5: Build disease mapping
    step5_ok = False
    if step3b_ok and step4_ok:
        step5_ok = build_disease_mapping()

    # Step 7: Enrich sparse signatures
    step7_ok = False
    if step3b_ok:
        step7_ok = enrich_sparse_signatures()

    # Step 8: Coverage report
    coverage_report()

    # Summary
    print("\n" + "=" * 60)
    print("  SETUP SUMMARY")
    print("=" * 60)
    steps = [
        ("DeepCE download", step1_ok),
        ("DeepCE processing", step2_ok),
        ("CREEDS download", step3_ok),
        ("CREEDS processing", step3b_ok),
        ("Drug ID mapping", step4_ok),
        ("Disease name mapping", step5_ok),
        ("Sparse signature enrichment", step7_ok),
    ]
    all_ok = True
    for name, ok in steps:
        status = "OK" if ok else "FAILED"
        print(f"  {name}: {status}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n  All steps completed successfully!")
        print("  You can now use the DGEM module.")
    else:
        print("\n  Some steps failed. Check the errors above.")
        print("  The DGEM module will work with partial data (graceful fallback).")

    print("=" * 60)
    return all_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
