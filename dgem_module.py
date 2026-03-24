"""
Module 5: Drug-Gene Expression Matching (DGEM)
===============================================
Scores drug-disease pairs by measuring how well a drug's gene expression
profile reverses a disease's expression signature.

This module runs alongside TxGNN and provides an orthogonal (functional)
scoring signal based on the Connectivity Map hypothesis: if a drug reverses
a disease's gene expression signature, it may treat that disease.

Data sources:
  - Drug profiles: DeepCE pre-computed predictions (11,179 DrugBank drugs)
  - Disease signatures: CREEDS database (555 human disease signatures)
  - Setup: Run scripts/setup_dgem.py first to download and process data
"""

import os
import json
import pickle
import numpy as np
from scipy import stats


class DGEMScorer:
    """Drug-Gene Expression Matching scorer for drug repurposing."""

    def __init__(self, data_folder="./data/dgem", cache_folder="./cache/dgem"):
        """
        Initialize the DGEM scorer.

        Args:
            data_folder: Path to processed DGEM data (from setup_dgem.py).
            cache_folder: Path for caching computed scores.
        """
        self.data_folder = data_folder
        self.cache_folder = cache_folder
        os.makedirs(cache_folder, exist_ok=True)

        self.drug_profiles = None       # {drugbank_id: np.array(n_genes,)}
        self.disease_signatures = None  # {disease_name: {vector, n_signatures, ...}}
        self.drug_id_mapping = None     # {name_to_id, idx_to_id, id_to_name}
        self.disease_name_mapping = None  # {txgnn_name: creeds_name}
        self.l1000_genes = None         # list of gene names
        self._initialized = False

        self._load_data()

    def _load_data(self):
        """Load all DGEM data files."""
        loaded = []
        failed = []

        # Drug profiles
        profiles_path = os.path.join(self.data_folder, "drug_profiles.pkl")
        if os.path.exists(profiles_path):
            with open(profiles_path, "rb") as f:
                self.drug_profiles = pickle.load(f)
            loaded.append(f"drug profiles ({len(self.drug_profiles)} drugs)")
        else:
            failed.append("drug_profiles.pkl")

        # Disease signatures
        sigs_path = os.path.join(self.data_folder, "disease_signatures.pkl")
        if os.path.exists(sigs_path):
            with open(sigs_path, "rb") as f:
                self.disease_signatures = pickle.load(f)
            loaded.append(f"disease signatures ({len(self.disease_signatures)} diseases)")
        else:
            failed.append("disease_signatures.pkl")

        # Drug ID mapping
        drug_map_path = os.path.join(self.data_folder, "drug_id_mapping.json")
        if os.path.exists(drug_map_path):
            with open(drug_map_path, "r") as f:
                self.drug_id_mapping = json.load(f)
            loaded.append("drug ID mapping")
        else:
            failed.append("drug_id_mapping.json")

        # Disease name mapping
        disease_map_path = os.path.join(self.data_folder, "disease_name_mapping.json")
        if os.path.exists(disease_map_path):
            with open(disease_map_path, "r") as f:
                self.disease_name_mapping = json.load(f)
            loaded.append(f"disease name mapping ({len(self.disease_name_mapping)} matches)")
        else:
            failed.append("disease_name_mapping.json")

        # L1000 gene list
        genes_path = os.path.join(self.data_folder, "l1000_genes.json")
        if os.path.exists(genes_path):
            with open(genes_path, "r") as f:
                self.l1000_genes = json.load(f)
            loaded.append(f"L1000 genes ({len(self.l1000_genes)})")
        else:
            failed.append("l1000_genes.json")

        # Report status
        if loaded:
            print(f"[DGEM] Loaded: {', '.join(loaded)}")
        if failed:
            print(f"[DGEM] Missing (run scripts/setup_dgem.py): {', '.join(failed)}")

        self._initialized = bool(self.drug_profiles and self.disease_signatures)
        if self._initialized:
            print("[DGEM] Module ready.")
        else:
            print("[DGEM] Module partially initialized. Some scoring may be unavailable.")

    def _get_drug_id(self, drug_name=None, drug_idx=None):
        """
        Get DrugBank ID for a drug.

        Args:
            drug_name: Drug name from TxGNN.
            drug_idx: Per-type drug index from TxGNN.

        Returns:
            DrugBank ID string or None.
        """
        if not self.drug_id_mapping:
            return None

        if drug_name and drug_name in self.drug_id_mapping.get("name_to_id", {}):
            return self.drug_id_mapping["name_to_id"][drug_name]

        if drug_idx is not None:
            return self.drug_id_mapping.get("idx_to_id", {}).get(str(drug_idx))

        return None

    def _get_disease_signature(self, disease_name):
        """
        Get disease expression signature vector.

        Args:
            disease_name: Disease name from TxGNN.

        Returns:
            numpy array of expression values, or None.
        """
        if not self.disease_signatures:
            return None

        disease_lower = disease_name.lower().strip()

        # 1. Direct lookup in CREEDS
        sig = None
        if disease_lower in self.disease_signatures:
            sig = self.disease_signatures[disease_lower]
        elif self.disease_name_mapping:
            # 2. Use pre-built mapping
            creeds_name = self.disease_name_mapping.get(disease_lower)
            if creeds_name and creeds_name in self.disease_signatures:
                sig = self.disease_signatures[creeds_name]

        if sig is None:
            return None

        vector = sig.get("vector")

        # If vector is a dict (gene->value) and we have L1000 gene list,
        # convert to aligned numpy array
        if isinstance(vector, dict) and self.l1000_genes:
            aligned = np.zeros(len(self.l1000_genes), dtype=np.float32)
            for i, gene in enumerate(self.l1000_genes):
                if gene in vector:
                    aligned[i] = vector[gene]
            return aligned

        # If vector is already numpy array, return as-is
        if isinstance(vector, np.ndarray):
            return vector

        # If vector is a dict but no L1000 gene list, try to align with
        # drug profiles (use the first drug profile's gene order)
        if isinstance(vector, dict):
            # Cannot align without gene list -- return None
            return None

        return None

    def _get_drug_profile(self, drug_name=None, drug_idx=None):
        """
        Get drug expression profile.

        Args:
            drug_name: Drug name from TxGNN.
            drug_idx: Per-type drug index from TxGNN.

        Returns:
            numpy array of expression values, or None.
        """
        if not self.drug_profiles:
            return None

        drugbank_id = self._get_drug_id(drug_name, drug_idx)
        if drugbank_id and drugbank_id in self.drug_profiles:
            return self.drug_profiles[drugbank_id]

        return None

    def compute_reversal_score(self, drug_profile, disease_signature):
        """
        Compute how well a drug reverses a disease's expression signature.

        Uses negative Pearson correlation mapped to [0, 1]:
          - r = -1 (perfect reversal) -> score = 1.0
          - r =  0 (no relationship)  -> score = 0.5
          - r = +1 (same direction)   -> score = 0.0

        For sparse disease signatures (>90% zeros), computes correlation
        using only the non-zero gene positions to avoid diluting the signal
        with uninformative zeros.

        Args:
            drug_profile: numpy array of drug-induced expression values.
            disease_signature: numpy array of disease expression values.

        Returns:
            Float score in [0, 1], or None if computation fails.
        """
        if drug_profile is None or disease_signature is None:
            return None

        # Ensure same length
        min_len = min(len(drug_profile), len(disease_signature))
        if min_len == 0:
            return None

        dp = drug_profile[:min_len].astype(np.float64)
        ds = disease_signature[:min_len].astype(np.float64)

        # Skip if either vector is all zeros or constant
        if np.std(dp) < 1e-10 or np.std(ds) < 1e-10:
            return 0.5  # No information -> neutral score

        # Sparse-aware scoring: if disease signature is very sparse,
        # use only the non-zero gene positions for correlation
        nonzero_mask = ds != 0
        sparsity = 1.0 - (np.count_nonzero(ds) / len(ds))

        if sparsity > 0.90 and np.sum(nonzero_mask) >= 10:
            # Focus on genes with known disease expression changes
            dp_focused = dp[nonzero_mask]
            ds_focused = ds[nonzero_mask]

            if np.std(dp_focused) < 1e-10 or np.std(ds_focused) < 1e-10:
                return 0.5

            pearson_r = np.corrcoef(dp_focused, ds_focused)[0, 1]
        else:
            # Standard full-vector correlation
            pearson_r = np.corrcoef(dp, ds)[0, 1]

        if np.isnan(pearson_r):
            return 0.5

        # Map from [-1, 1] to [0, 1] where negative correlation = high score
        reversal_score = (-pearson_r + 1.0) / 2.0
        return float(reversal_score)

    def compute_enrichment_score(self, drug_profile, up_genes, down_genes):
        """
        CMap-style enrichment score using KS test.

        Checks if disease up-genes are suppressed by the drug (enriched at
        bottom of drug-ranked gene list) and vice versa.

        Args:
            drug_profile: numpy array of drug expression values.
            up_genes: list of gene indices that are upregulated in disease.
            down_genes: list of gene indices that are downregulated in disease.

        Returns:
            Float score in [0, 1].
        """
        if drug_profile is None or (not up_genes and not down_genes):
            return None

        n_genes = len(drug_profile)
        # Rank genes by drug expression (ascending = most suppressed first)
        ranked = np.argsort(drug_profile)

        scores = []

        if up_genes:
            # Disease up-genes should be at the BOTTOM of drug ranking (suppressed)
            positions = np.array([np.where(ranked == g)[0][0] for g in up_genes
                                  if g < n_genes])
            if len(positions) > 0:
                # KS test: are these positions enriched at high ranks (bottom)?
                ks_stat, _ = stats.ks_2samp(positions, np.arange(n_genes))
                # Check direction: we want up_genes to be suppressed
                mean_pos = np.mean(positions) / n_genes
                up_score = ks_stat if mean_pos < 0.5 else -ks_stat
                scores.append(up_score)

        if down_genes:
            # Disease down-genes should be at the TOP of drug ranking (activated)
            positions = np.array([np.where(ranked == g)[0][0] for g in down_genes
                                  if g < n_genes])
            if len(positions) > 0:
                ks_stat, _ = stats.ks_2samp(positions, np.arange(n_genes))
                mean_pos = np.mean(positions) / n_genes
                down_score = ks_stat if mean_pos > 0.5 else -ks_stat
                scores.append(down_score)

        if not scores:
            return None

        # Average and map to [0, 1]
        raw = np.mean(scores)
        return float((raw + 1.0) / 2.0)

    def score_drugs_for_disease(self, disease_name, drug_names=None,
                                 drug_indices=None, top_k=20):
        """
        Score drugs against a disease using gene expression reversal.

        Args:
            disease_name: Disease name from TxGNN.
            drug_names: Optional list of drug names to score (from TxGNN predictions).
            drug_indices: Optional list of (drug_idx, drug_name) tuples.
            top_k: Number of top results to return.

        Returns:
            Dict mapping drug_name -> dgem_score, or empty dict if unavailable.
        """
        if not self._initialized:
            return {}

        # Get disease signature
        disease_sig = self._get_disease_signature(disease_name)
        if disease_sig is None:
            return {}

        results = {}

        if drug_names:
            for drug_name in drug_names:
                drug_profile = self._get_drug_profile(drug_name=drug_name)
                if drug_profile is not None:
                    score = self.compute_reversal_score(drug_profile, disease_sig)
                    if score is not None:
                        results[drug_name] = score

        elif drug_indices:
            for drug_idx, drug_name in drug_indices:
                drug_profile = self._get_drug_profile(drug_idx=drug_idx)
                if drug_profile is not None:
                    score = self.compute_reversal_score(drug_profile, disease_sig)
                    if score is not None:
                        results[drug_name] = score

        return results

    def is_available_for_drug(self, drug_name=None, drug_idx=None):
        """Check if DGEM data exists for a given drug."""
        return self._get_drug_profile(drug_name=drug_name, drug_idx=drug_idx) is not None

    def is_available_for_disease(self, disease_name):
        """Check if a disease signature is available and usable."""
        sig = self._get_disease_signature(disease_name)
        if sig is None:
            return False
        if isinstance(sig, np.ndarray):
            return True
        return False

    def get_coverage_stats(self):
        """Report DGEM data coverage statistics."""
        stats_dict = {
            "initialized": self._initialized,
            "n_drug_profiles": len(self.drug_profiles) if self.drug_profiles else 0,
            "n_disease_signatures": len(self.disease_signatures) if self.disease_signatures else 0,
            "n_disease_mappings": len(self.disease_name_mapping) if self.disease_name_mapping else 0,
            "n_genes": len(self.l1000_genes) if self.l1000_genes else 0,
        }
        return stats_dict


# ─────────────────────────────────────────────────────────────
# Score combination utilities
# ─────────────────────────────────────────────────────────────

def combine_scores(gnn_score, dgem_score, w_gnn=0.6, w_dgem=0.4):
    """
    Combine GNN and DGEM scores using weighted average.

    Args:
        gnn_score: TxGNN score (0-1).
        dgem_score: DGEM reversal score (0-1), or None.
        w_gnn: Weight for GNN score (default 0.6).
        w_dgem: Weight for DGEM score (default 0.4).

    Returns:
        Combined score (float).
    """
    if dgem_score is None:
        return gnn_score
    if gnn_score is None:
        return dgem_score
    return w_gnn * gnn_score + w_dgem * dgem_score


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing DGEM Module...")
    scorer = DGEMScorer()

    stats = scorer.get_coverage_stats()
    print(f"\nCoverage stats: {json.dumps(stats, indent=2)}")

    # Test with a known pair if data is available
    if scorer._initialized:
        test_disease = "fragile x syndrome"
        test_drug = "Sulindac"

        print(f"\nTesting: {test_drug} -> {test_disease}")
        print(f"  Drug available: {scorer.is_available_for_drug(drug_name=test_drug)}")
        print(f"  Disease available: {scorer.is_available_for_disease(test_disease)}")

        scores = scorer.score_drugs_for_disease(
            test_disease, drug_names=[test_drug]
        )
        if scores:
            print(f"  Reversal score: {scores[test_drug]:.4f}")
        else:
            print("  No score computed (data missing for this pair)")
    else:
        print("\nDGEM not initialized. Run: python scripts/setup_dgem.py")
