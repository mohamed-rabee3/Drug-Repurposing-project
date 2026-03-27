"""
Build a robust FXS+Autism meta-analysis disease signature
==========================================================
Based on the DREAM-RD approach (Agastheeswaramoorthy & Sevilimedu 2020):
Combine FXS and Autism GEO datasets to capture shared molecular pathways
(especially Wnt signaling) that are dysregulated in both conditions.

Datasets used (from CREEDS, matching DREAM-RD paper):
  - GSE7329:  FXS + Autism (blood, human)
  - GSE25507: Autism spectrum disorder (blood, human)
  - GSE29691: Autism spectrum disorder (blood, human)
  - GSE62632: Autism spectrum disorder (blood, human)

Meta-analysis method:
  1. Build per-signature L1000-aligned vectors from CREEDS DEGs
  2. For each gene, compute direction consistency (vote across signatures)
  3. Weight by mean effect size × consistency fraction
  4. Filter: keep only genes with ≥30% of signatures reporting non-zero value
  5. Replace the existing 'fragile x syndrome' entry in disease_signatures.pkl

Run: python scripts/build_fxs_meta_signature.py
"""

import os
import sys
import json
import pickle
import numpy as np

# Paths
DGEM_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data", "dgem")
CREEDS_PATH = os.path.join(DGEM_FOLDER, "creeds_raw.json")
SIGS_PATH = os.path.join(DGEM_FOLDER, "disease_signatures.pkl")
GENES_PATH = os.path.join(DGEM_FOLDER, "l1000_genes.json")

# DREAM-RD target datasets
DREAM_RD_GEOS = {"GSE7329", "GSE29691", "GSE62632", "GSE25507"}


def build_meta_signature():
    """Build FXS+Autism meta-analysis signature and update disease_signatures.pkl."""

    # Load required files
    if not os.path.exists(CREEDS_PATH):
        print("[ERROR] CREEDS raw data not found. Run setup_dgem.py first.")
        return False

    if not os.path.exists(GENES_PATH):
        print("[ERROR] L1000 gene list not found. Run setup_dgem.py first.")
        return False

    with open(CREEDS_PATH, "r", encoding="utf-8") as f:
        creeds = json.load(f)
    with open(GENES_PATH, "r") as f:
        l1000_genes = json.load(f)

    gene_to_idx = {g: i for i, g in enumerate(l1000_genes)}
    n_genes = len(l1000_genes)

    # Collect FXS + Autism human signatures from DREAM-RD datasets
    matching_sigs = [
        s for s in creeds
        if s.get("organism", "").lower() == "human"
        and s.get("geo_id", "") in DREAM_RD_GEOS
        and ("fragile" in s.get("disease_name", "").lower()
             or "autis" in s.get("disease_name", "").lower())
    ]

    if not matching_sigs:
        print("[ERROR] No matching FXS/Autism signatures found in CREEDS data.")
        return False

    print(f"[META] Found {len(matching_sigs)} FXS+Autism signatures from "
          f"{len(DREAM_RD_GEOS)} DREAM-RD datasets:")
    geo_counts = {}
    for s in matching_sigs:
        geo = s["geo_id"]
        geo_counts[geo] = geo_counts.get(geo, 0) + 1
    for geo, count in sorted(geo_counts.items()):
        print(f"  {geo}: {count} signatures")

    # Build per-signature vectors aligned to L1000
    vectors = []
    for s in matching_sigs:
        vec = np.zeros(n_genes, dtype=np.float64)
        for gene_name, cd_value in s.get("up_genes", []):
            if gene_name in gene_to_idx:
                vec[gene_to_idx[gene_name]] = float(cd_value)
        for gene_name, cd_value in s.get("down_genes", []):
            if gene_name in gene_to_idx:
                vec[gene_to_idx[gene_name]] = float(cd_value)
        vectors.append(vec)

    vectors = np.array(vectors)  # (n_sigs, n_genes)
    n_sigs = len(vectors)

    # Meta-analysis: robust direction-consistent effect size
    # For each gene:
    #   - Count how many signatures show up/down regulation
    #   - Determine majority direction
    #   - Weight by mean absolute effect size × consistency fraction
    #   - Filter genes with too few votes
    signs = np.sign(vectors)
    vote_up = np.sum(signs > 0, axis=0)
    vote_down = np.sum(signs < 0, axis=0)
    vote_total = vote_up + vote_down  # non-zero votes per gene

    majority_up = vote_up > vote_down
    consistency = np.where(
        majority_up, vote_up, vote_down
    ) / np.maximum(vote_total, 1)

    # Require at least 30% of signatures to have a non-zero value
    min_votes = max(3, int(n_sigs * 0.3))
    keep_mask = vote_total >= min_votes

    # Effect size: mean absolute value across all signatures
    mean_effect = np.mean(np.abs(vectors), axis=0)
    direction = np.where(majority_up, 1.0, -1.0)

    # Final meta-signature
    meta_vec = (direction * mean_effect * consistency).astype(np.float32)
    meta_vec[~keep_mask] = 0.0

    n_nonzero = int(np.count_nonzero(meta_vec))
    print(f"\n[META] Meta-signature: {n_nonzero} informative genes "
          f"(from {n_genes} L1000 genes)")
    print(f"[META] Min votes threshold: {min_votes}/{n_sigs}")

    # Report top dysregulated genes
    abs_meta = np.abs(meta_vec)
    top_idx = np.argsort(abs_meta)[::-1][:20]
    print("\n[META] Top 20 most dysregulated genes in meta-signature:")
    for i, idx in enumerate(top_idx):
        if meta_vec[idx] == 0:
            break
        gene = l1000_genes[idx]
        val = meta_vec[idx]
        direction_str = "UP" if val > 0 else "DOWN"
        n_agree = int(vote_up[idx] if val > 0 else vote_down[idx])
        print(f"  {i+1:2d}. {gene:12s} {direction_str:4s} "
              f"effect={abs(val):.4f}  consistency={n_agree}/{int(vote_total[idx])}")

    # Update disease_signatures.pkl
    if not os.path.exists(SIGS_PATH):
        print("[ERROR] disease_signatures.pkl not found.")
        return False

    with open(SIGS_PATH, "rb") as f:
        disease_sigs = pickle.load(f)

    # Store old signature info for comparison
    old_entry = disease_sigs.get("fragile x syndrome", {})
    old_nonzero = (int(np.count_nonzero(old_entry["vector"]))
                   if "vector" in old_entry
                   and isinstance(old_entry["vector"], np.ndarray)
                   else 0)
    old_geos = old_entry.get("geo_ids", [])

    # Replace with meta-signature
    all_geo_ids = sorted(set(s["geo_id"] for s in matching_sigs))
    disease_sigs["fragile x syndrome"] = {
        "vector": meta_vec,
        "n_signatures": n_sigs,
        "geo_ids": all_geo_ids,
        "avg_l1000_matched": n_nonzero,
        "meta_analysis": {
            "method": "direction_consistent_effect_size",
            "source_datasets": list(DREAM_RD_GEOS),
            "n_source_signatures": n_sigs,
            "min_vote_threshold": min_votes,
            "reference": "Agastheeswaramoorthy & Sevilimedu 2020 (DREAM-RD)",
        },
    }

    with open(SIGS_PATH, "wb") as f:
        pickle.dump(disease_sigs, f, protocol=4)

    print(f"\n[META] Updated disease_signatures.pkl:")
    print(f"  Old: {old_nonzero} nonzero genes from {old_geos}")
    print(f"  New: {n_nonzero} nonzero genes from {all_geo_ids}")
    print(f"  Method: direction-consistent effect size meta-analysis")
    print(f"\n[META] Done! FXS signature now uses {n_sigs} combined "
          f"FXS+Autism signatures.")
    return True


if __name__ == "__main__":
    success = build_meta_signature()
    sys.exit(0 if success else 1)
