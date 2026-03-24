"""
Module 7: Network Proximity Scoring
=====================================
Scores drug-disease pairs by shortest-path distance between drug targets
and disease-associated proteins in the protein-protein interaction network.

Based on: Guney et al. 2016 "Network-based in silico drug efficacy screening"

Data source: TxGNN knowledge graph (pre-extracted by scripts/setup_network.py)
"""

import os
import hashlib
import pickle
import numpy as np
import networkx as nx


class ProximityScorer:
    """Score drugs by PPI network proximity to disease proteins."""

    def __init__(self, data_folder="./data/network",
                 cache_folder="./cache/proximity"):
        self.data_folder = data_folder
        self.cache_folder = cache_folder
        os.makedirs(cache_folder, exist_ok=True)

        self.drug_targets = None       # {drug_name_lower: set[protein]}
        self.disease_proteins = None   # {disease_name_lower: set[protein]}
        self.ppi_graph = None          # networkx.Graph
        self._initialized = False

        self._load_data()

    def _load_data(self):
        loaded = []
        failed = []

        for attr, fname in [("drug_targets", "drug_targets.pkl"),
                             ("disease_proteins", "disease_proteins.pkl"),
                             ("ppi_graph", "ppi_graph.pkl")]:
            path = os.path.join(self.data_folder, fname)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    setattr(self, attr, pickle.load(f))
                loaded.append(fname)
            else:
                failed.append(fname)

        if loaded:
            print(f"[PROXIMITY] Loaded: {', '.join(loaded)}")
        if failed:
            print(f"[PROXIMITY] Missing: {', '.join(failed)}")

        self._initialized = bool(
            self.drug_targets and self.disease_proteins and self.ppi_graph
        )

        if self._initialized:
            print(f"[PROXIMITY] Ready. PPI: {self.ppi_graph.number_of_nodes()} "
                  f"nodes, {self.ppi_graph.number_of_edges()} edges")
        else:
            print("[PROXIMITY] Module not initialized.")

    def _compute_disease_distances(self, disease_name):
        """
        BFS from each disease protein with cutoff=4.
        Returns {protein: min_distance_to_any_disease_protein}.
        """
        disease_lower = disease_name.lower().strip()
        disease_prots = self.disease_proteins.get(disease_lower, set())
        if not disease_prots:
            return None

        # Check cache
        cache_key = hashlib.md5(disease_lower.encode()).hexdigest()
        cache_path = os.path.join(self.cache_folder, f"{cache_key}.pkl")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        # BFS from each disease protein, merge min distances
        distances = {}  # protein -> min distance to any disease protein
        cutoff = 4

        for prot in disease_prots:
            if prot not in self.ppi_graph:
                continue
            lengths = nx.single_source_shortest_path_length(
                self.ppi_graph, prot, cutoff=cutoff
            )
            for node, dist in lengths.items():
                if node not in distances or dist < distances[node]:
                    distances[node] = dist

        # Cache result
        with open(cache_path, "wb") as f:
            pickle.dump(distances, f, protocol=4)

        return distances

    def score_drugs_for_disease(self, disease_name, drug_names=None, top_k=20):
        """
        Score drugs by network proximity to disease proteins.

        Args:
            disease_name: Disease name from TxGNN.
            drug_names: List of drug names to score. If None, score all.
            top_k: Number of top results to return.

        Returns:
            Dict mapping drug_name -> proximity_score (0-1).
        """
        if not self._initialized:
            return {}

        distances = self._compute_disease_distances(disease_name)
        if not distances:
            return {}

        if drug_names is None:
            drug_names = list(self.drug_targets.keys())

        cutoff = 4
        default_dist = cutoff + 1

        raw_scores = {}
        for drug_name in drug_names:
            drug_lower = drug_name.lower().strip()
            targets = self.drug_targets.get(drug_lower)
            if not targets:
                continue

            # For each drug target, find min distance to disease proteins
            target_dists = []
            for t in targets:
                if t in distances:
                    target_dists.append(distances[t])
                else:
                    target_dists.append(default_dist)

            if not target_dists:
                continue

            mean_dist = np.mean(target_dists)
            # Convert distance to score: closer = higher
            raw_scores[drug_name] = 1.0 / (1.0 + mean_dist)

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
            "ppi_nodes": self.ppi_graph.number_of_nodes() if self.ppi_graph else 0,
            "ppi_edges": self.ppi_graph.number_of_edges() if self.ppi_graph else 0,
        }


if __name__ == "__main__":
    import json
    scorer = ProximityScorer()
    print(f"\nCoverage: {json.dumps(scorer.get_coverage_stats(), indent=2)}")

    if scorer._initialized:
        disease = "fragile x syndrome"
        print(f"\nScoring all drugs for '{disease}'...")
        scores = scorer.score_drugs_for_disease(disease)
        print(f"Scored {len(scores)} drugs")

        if scores:
            top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:20]
            print(f"\nTop 20 by proximity score:")
            for i, (drug, score) in enumerate(top):
                print(f"  {i+1}. {drug}: {score:.4f}")

            sulindac_score = scores.get("Sulindac") or scores.get("sulindac")
            if sulindac_score is not None:
                rank = sum(1 for s in scores.values() if s > sulindac_score) + 1
                print(f"\nSulindac: score={sulindac_score:.4f}, "
                      f"rank={rank}/{len(scores)}")
