"""
Module 6: Pathway-Level Scoring
================================
Scores drug-disease pairs by testing whether a drug's protein targets
are enriched in disease-relevant biological pathways (Fisher's exact test).

Data source: TxGNN knowledge graph (pre-extracted by scripts/setup_network.py)
"""

import os
import pickle
import numpy as np
from scipy import stats


class PathwayScorer:
    """Score drugs by pathway overlap with disease-associated proteins."""

    def __init__(self, data_folder="./data/network",
                 cache_folder="./cache/pathway"):
        self.data_folder = data_folder
        self.cache_folder = cache_folder
        os.makedirs(cache_folder, exist_ok=True)

        self.drug_targets = None       # {drug_name_lower: set[protein]}
        self.disease_proteins = None   # {disease_name_lower: set[protein]}
        self.pathway_proteins = None   # {pathway_name: set[protein]}
        self.protein_pathways = None   # {protein_name: set[pathway]}
        self._all_proteins = None      # set of all proteins in any pathway
        self._initialized = False

        self._load_data()

    def _load_data(self):
        loaded = []
        failed = []

        files = {
            "drug_targets": "drug_targets.pkl",
            "disease_proteins": "disease_proteins.pkl",
            "pathway_proteins": "pathway_proteins.pkl",
            "protein_pathways": "protein_pathways.pkl",
        }

        for attr, fname in files.items():
            path = os.path.join(self.data_folder, fname)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    setattr(self, attr, pickle.load(f))
                loaded.append(fname)
            else:
                failed.append(fname)

        if loaded:
            print(f"[PATHWAY] Loaded: {', '.join(loaded)}")
        if failed:
            print(f"[PATHWAY] Missing (run scripts/setup_network.py): "
                  f"{', '.join(failed)}")

        self._initialized = bool(
            self.drug_targets and self.disease_proteins
            and self.pathway_proteins and self.protein_pathways
        )

        if self._initialized:
            # Build universe of all proteins in any pathway
            self._all_proteins = set()
            for proteins in self.pathway_proteins.values():
                self._all_proteins.update(proteins)
            print(f"[PATHWAY] Ready. {len(self.drug_targets)} drugs, "
                  f"{len(self.disease_proteins)} diseases, "
                  f"{len(self.pathway_proteins)} pathways, "
                  f"{len(self._all_proteins)} unique proteins")
        else:
            print("[PATHWAY] Module not initialized.")

    def score_drugs_for_disease(self, disease_name, drug_names=None, top_k=20):
        """
        Score drugs by pathway enrichment with disease proteins.

        Args:
            disease_name: Disease name from TxGNN.
            drug_names: List of drug names to score. If None, score all.
            top_k: Number of top results to return.

        Returns:
            Dict mapping drug_name -> pathway_score (0-1).
        """
        if not self._initialized:
            return {}

        disease_lower = disease_name.lower().strip()
        disease_prots = self.disease_proteins.get(disease_lower)
        if not disease_prots:
            return {}

        # Find disease-relevant pathways
        disease_pathways = {}  # pathway -> set of disease proteins in it
        for prot in disease_prots:
            for pw in self.protein_pathways.get(prot, set()):
                if pw not in disease_pathways:
                    disease_pathways[pw] = set()
                disease_pathways[pw].add(prot)

        if not disease_pathways:
            return {}

        universe_size = len(self._all_proteins)

        # Score each drug
        if drug_names is None:
            drug_names = list(self.drug_targets.keys())

        raw_scores = {}
        for drug_name in drug_names:
            drug_lower = drug_name.lower().strip()
            targets = self.drug_targets.get(drug_lower)
            if not targets:
                continue

            best_pval = 1.0
            for pw_name, pw_disease_prots in disease_pathways.items():
                pw_all_prots = self.pathway_proteins[pw_name]

                # Fisher's exact test:
                # Are drug targets enriched in this pathway?
                overlap = len(targets & pw_all_prots)
                if overlap == 0:
                    continue

                drug_not_pw = len(targets) - overlap
                pw_not_drug = len(pw_all_prots) - overlap
                neither = universe_size - len(targets) - len(pw_all_prots) + overlap

                table = [[overlap, drug_not_pw],
                         [pw_not_drug, max(neither, 0)]]
                _, pval = stats.fisher_exact(table, alternative="greater")

                if pval < best_pval:
                    best_pval = pval

            if best_pval < 1.0:
                raw_scores[drug_name] = -np.log10(best_pval + 1e-300)

        if not raw_scores:
            return {}

        # Min-max normalize to [0, 1]
        values = list(raw_scores.values())
        min_val = min(values)
        max_val = max(values)
        score_range = max_val - min_val

        results = {}
        for drug, raw in raw_scores.items():
            if score_range > 0:
                results[drug] = (raw - min_val) / score_range
            else:
                results[drug] = 1.0

        return results

    def is_available_for_drug(self, drug_name):
        if not self.drug_targets:
            return False
        return drug_name.lower().strip() in self.drug_targets

    def is_available_for_disease(self, disease_name):
        if not self.disease_proteins:
            return False
        return disease_name.lower().strip() in self.disease_proteins

    def get_coverage_stats(self):
        return {
            "initialized": self._initialized,
            "n_drugs_with_targets": len(self.drug_targets) if self.drug_targets else 0,
            "n_diseases_with_proteins": len(self.disease_proteins) if self.disease_proteins else 0,
            "n_pathways": len(self.pathway_proteins) if self.pathway_proteins else 0,
            "n_proteins_in_pathways": len(self._all_proteins) if self._all_proteins else 0,
        }


if __name__ == "__main__":
    import json
    scorer = PathwayScorer()
    print(f"\nCoverage: {json.dumps(scorer.get_coverage_stats(), indent=2)}")

    if scorer._initialized:
        disease = "fragile x syndrome"
        print(f"\nScoring all drugs for '{disease}'...")
        scores = scorer.score_drugs_for_disease(disease)
        print(f"Scored {len(scores)} drugs")

        if scores:
            top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:20]
            print(f"\nTop 20 by pathway score:")
            for i, (drug, score) in enumerate(top):
                print(f"  {i+1}. {drug}: {score:.4f}")

            # Check Sulindac
            sulindac_score = scores.get("Sulindac") or scores.get("sulindac")
            if sulindac_score is not None:
                rank = sum(1 for s in scores.values() if s > sulindac_score) + 1
                print(f"\nSulindac: score={sulindac_score:.4f}, "
                      f"rank={rank}/{len(scores)}")
