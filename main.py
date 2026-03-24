"""
Orchestrator: Connects TxGNN (graph) + GPT-OSS 20B (language) modules
=====================================================================
This is the main entry point for the drug repurposing system.
"""

import torch_cuda_ld_path

torch_cuda_ld_path.apply()

import json
import os
import time
from datetime import datetime
from graph_module import DrugRepurposingGNN
from nlp_module import BiomedicalNLP
from explain_module import RepurposingExplainer
from eval_module import RepurposingEvaluator
from dgem_module import DGEMScorer


class DrugRepurposingSystem:
    """Main system that orchestrates GNN + NLP for drug repurposing."""

    def __init__(self, data_folder="./data", model_folder="./models",
                 output_folder="./outputs", enable_dgem=True,
                 enable_pathway=True, enable_proximity=True,
                 enable_literature=True):
        """
        Initialize all modules.

        Args:
            data_folder: Where TxGNN knowledge graph data is stored.
            model_folder: Where trained model checkpoints are saved.
            output_folder: Where results are written.
            enable_dgem: If True, initialize DGEM gene expression module.
            enable_pathway: If True, initialize pathway scoring module.
            enable_proximity: If True, initialize proximity scoring module.
            enable_literature: If True, initialize literature mining module.
        """
        self.output_folder = output_folder
        try:
            os.makedirs(output_folder, exist_ok=True)
        except (FileExistsError, OSError):
            pass

        print("=" * 60)
        print("  AI DRUG REPURPOSING SYSTEM")
        print("  TxGNN + DGEM + Pathway + Proximity + Literature")
        print("  GPT-OSS 20B (Groq)")
        print("=" * 60)

        # Initialize modules
        print("\n[SYSTEM] Initializing GNN module...")
        self.gnn = DrugRepurposingGNN(data_folder, model_folder)

        print("\n[SYSTEM] Initializing NLP module...")
        self.nlp = BiomedicalNLP()

        # Initialize DGEM module
        self.dgem = None
        if enable_dgem:
            dgem_folder = os.path.join(data_folder, "dgem")
            if os.path.exists(dgem_folder):
                print("\n[SYSTEM] Initializing DGEM module...")
                self.dgem = DGEMScorer(data_folder=dgem_folder)
            else:
                print("\n[SYSTEM] DGEM data not found. "
                      "Run: python scripts/setup_dgem.py")

        # Initialize Pathway module
        self.pathway = None
        network_folder = os.path.join(data_folder, "network")
        if enable_pathway and os.path.exists(network_folder):
            print("\n[SYSTEM] Initializing Pathway module...")
            from pathway_module import PathwayScorer
            self.pathway = PathwayScorer(data_folder=network_folder)
        elif enable_pathway:
            print("\n[SYSTEM] Network data not found. "
                  "Run: python scripts/setup_network.py")

        # Initialize Proximity module
        self.proximity = None
        if enable_proximity and os.path.exists(network_folder):
            print("\n[SYSTEM] Initializing Proximity module...")
            from proximity_module import ProximityScorer
            self.proximity = ProximityScorer(data_folder=network_folder)
        elif enable_proximity:
            print("\n[SYSTEM] Network data not found. "
                  "Run: python scripts/setup_network.py")

        # Initialize Literature module
        self.literature = None
        if enable_literature:
            print("\n[SYSTEM] Initializing Literature module...")
            from literature_module import LiteratureScorer
            self.literature = LiteratureScorer(nlp_module=self.nlp)

        # Initialize explainability and evaluation modules
        self.explainer = RepurposingExplainer(self.nlp, output_folder)
        self.evaluator = RepurposingEvaluator(output_folder)

        print("\n[SYSTEM] System ready!")

    def setup_gnn(self, split="random", train=True, epochs=500):
        """
        Set up the GNN: load KG, initialize, and optionally train.

        Args:
            split: "random" for easy testing, "complex_disease" for real eval.
            train: If True, train from scratch. If False, try to load saved model.
            epochs: Number of fine-tuning epochs.
        """
        self.gnn.load_knowledge_graph(split=split)
        self.gnn.initialize_model()

        if train:
            self.gnn.train(finetune_epochs=epochs)
        else:
            try:
                self.gnn.load_model()
            except FileNotFoundError:
                print("[SYSTEM] No saved model found. Training new model...")
                self.gnn.train(finetune_epochs=epochs)

    def repurpose(self, disease_name, top_k=10, explain_top=5):
        """
        Full drug repurposing pipeline for a disease.

        Produces separate ranked lists from each scoring module:
          - TxGNN: GNN embedding similarity
          - DGEM: gene expression reversal
          - Pathway: pathway enrichment (Fisher's exact test)
          - Proximity: PPI network distance
          - Literature: PubMed co-occurrence + LLM relevance

        Args:
            disease_name: e.g., "Alzheimer disease"
            top_k: Number of drug candidates from each method.
            explain_top: Number of top drugs to explain via NLP.

        Returns:
            Dict with complete results.
        """
        use_dgem = self.dgem is not None and self.dgem._initialized
        use_pathway = self.pathway is not None and self.pathway._initialized
        use_proximity = self.proximity is not None and self.proximity._initialized
        use_literature = self.literature is not None

        # Count active steps
        n_steps = 3  # NLP context + GNN + report
        if explain_top > 0:
            n_steps += 1
        if use_dgem:
            n_steps += 1
        if use_pathway:
            n_steps += 1
        if use_proximity:
            n_steps += 1
        if use_literature:
            n_steps += 1

        modules = ["TxGNN"]
        if use_dgem:
            modules.append("DGEM")
        if use_pathway:
            modules.append("Pathway")
        if use_proximity:
            modules.append("Proximity")
        if use_literature:
            modules.append("Literature")

        print(f"\n{'='*60}")
        print(f"  REPURPOSING DRUGS FOR: {disease_name}")
        print(f"  Modules: {' + '.join(modules)}")
        print(f"{'='*60}")
        start_time = time.time()

        step = 0

        # Step: Disease context
        step += 1
        print(f"\n[Step {step}/{n_steps}] Getting disease context "
              "from GPT-OSS 20B...")
        disease_context = self.nlp.enrich_disease_context(disease_name)

        # Step: GNN predictions
        step += 1
        print(f"\n[Step {step}/{n_steps}] Predicting top {top_k} "
              "drug candidates with TxGNN...")
        try:
            disease_idx = self.gnn.find_disease_idx(disease_name)
            gnn_predictions = self.gnn.predict_drugs_for_disease(
                disease_idx=disease_idx, top_k=top_k
            )
        except (RuntimeError, ValueError) as e:
            print(f"[SYSTEM] GNN prediction failed: {e}")
            gnn_predictions = []

        # Step: DGEM scoring
        dgem_predictions = []
        dgem_drugs_scored = 0
        if use_dgem:
            step += 1
            print(f"\n[Step {step}/{n_steps}] DGEM gene expression "
                  "reversal scoring...")
            all_drug_names = list(
                self.dgem.drug_id_mapping.get("name_to_id", {}).keys()
            ) if self.dgem.drug_id_mapping else []
            dgem_scores = self.dgem.score_drugs_for_disease(
                disease_name, drug_names=all_drug_names
            )
            if dgem_scores:
                dgem_drugs_scored = len(dgem_scores)
                sorted_dgem = sorted(
                    dgem_scores.items(), key=lambda x: x[1], reverse=True
                )
                for rank, (drug, score) in enumerate(sorted_dgem[:top_k]):
                    dgem_predictions.append({
                        "drug": drug, "score": round(score, 6),
                        "rank": rank + 1,
                    })
                print(f"[DGEM] Scored {dgem_drugs_scored} drugs")

        # Step: Pathway scoring
        pathway_predictions = []
        pathway_drugs_scored = 0
        if use_pathway:
            step += 1
            print(f"\n[Step {step}/{n_steps}] Pathway enrichment scoring...")
            pw_scores = self.pathway.score_drugs_for_disease(disease_name)
            if pw_scores:
                pathway_drugs_scored = len(pw_scores)
                sorted_pw = sorted(
                    pw_scores.items(), key=lambda x: x[1], reverse=True
                )
                for rank, (drug, score) in enumerate(sorted_pw[:top_k]):
                    pathway_predictions.append({
                        "drug": drug, "score": round(score, 6),
                        "rank": rank + 1,
                    })
                print(f"[PATHWAY] Scored {pathway_drugs_scored} drugs")
            else:
                print("[PATHWAY] No pathway data for this disease.")

        # Step: Proximity scoring
        proximity_predictions = []
        proximity_drugs_scored = 0
        if use_proximity:
            step += 1
            print(f"\n[Step {step}/{n_steps}] PPI network proximity scoring...")
            prox_scores = self.proximity.score_drugs_for_disease(disease_name)
            if prox_scores:
                proximity_drugs_scored = len(prox_scores)
                sorted_prox = sorted(
                    prox_scores.items(), key=lambda x: x[1], reverse=True
                )
                for rank, (drug, score) in enumerate(sorted_prox[:top_k]):
                    proximity_predictions.append({
                        "drug": drug, "score": round(score, 6),
                        "rank": rank + 1,
                    })
                print(f"[PROXIMITY] Scored {proximity_drugs_scored} drugs")
            else:
                print("[PROXIMITY] No proximity data for this disease.")

        # Step: Literature scoring (pre-filtered to top candidates)
        literature_predictions = []
        literature_drugs_scored = 0
        if use_literature:
            step += 1
            print(f"\n[Step {step}/{n_steps}] Literature mining (PubMed)...")

            # Collect top-50 from each module for pre-filtering
            lit_candidates = set()
            for pred_list in [gnn_predictions, dgem_predictions,
                              pathway_predictions, proximity_predictions]:
                for p in pred_list[:50]:
                    lit_candidates.add(p["drug"])

            if lit_candidates:
                lit_scores = self.literature.score_drugs_for_disease(
                    disease_name, drug_names=list(lit_candidates)
                )
                if lit_scores:
                    literature_drugs_scored = len(lit_scores)
                    sorted_lit = sorted(
                        lit_scores.items(), key=lambda x: x[1], reverse=True
                    )
                    details = self.literature.get_details()
                    for rank, (drug, score) in enumerate(sorted_lit[:top_k]):
                        entry = {
                            "drug": drug, "score": round(score, 6),
                            "rank": rank + 1,
                        }
                        info = details.get(drug, {})
                        entry["pubmed_count"] = info.get("count", 0)
                        literature_predictions.append(entry)

        # Step: Explain top predictions (TxGNN + DGEM top 5)
        if explain_top > 0:
            step += 1
            print(f"\n[Step {step}/{n_steps}] Explaining top {explain_top} "
                  "candidates with GPT-OSS 20B...")
            # Collect unique drug names from TxGNN and DGEM top-5
            explain_drugs = []
            seen = set()
            for pred_list in [gnn_predictions, dgem_predictions]:
                for p in pred_list[:explain_top]:
                    name = p["drug"]
                    if name.lower() not in seen:
                        seen.add(name.lower())
                        explain_drugs.append(name)
            explanations = (
                self.nlp.batch_explain(disease_name, explain_drugs)
                if explain_drugs else {}
            )
        else:
            explanations = {}

        # Step: Report
        step += 1
        print(f"\n[Step {step}/{n_steps}] Generating comprehensive report...")
        if gnn_predictions:
            report = self.nlp.generate_report(disease_name, gnn_predictions)
        else:
            report = self.nlp.generate_report(
                disease_name, [{"drug": "N/A", "score": 0}]
            )

        # Build results
        elapsed = time.time() - start_time
        results = {
            "disease": disease_name,
            "disease_context": disease_context,
            "txgnn_predictions": gnn_predictions,
            "dgem_predictions": dgem_predictions,
            "pathway_predictions": pathway_predictions,
            "proximity_predictions": proximity_predictions,
            "literature_predictions": literature_predictions,
            "explanations": explanations,
            "report": report,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "modules": modules,
                "top_k": top_k,
                "processing_time_seconds": round(elapsed, 2),
                "dgem_drugs_scored": dgem_drugs_scored,
                "pathway_drugs_scored": pathway_drugs_scored,
                "proximity_drugs_scored": proximity_drugs_scored,
                "literature_drugs_scored": literature_drugs_scored,
            }
        }

        # Save results
        safe_name = disease_name.replace(" ", "_").lower()
        output_path = os.path.join(
            self.output_folder,
            f"repurposing_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n{'='*60}")
        print(f"  RESULTS SAVED: {output_path}")
        print(f"  Processing time: {elapsed:.1f} seconds")
        print(f"{'='*60}")

        # Print summaries
        summaries = [
            ("TxGNN", gnn_predictions),
            ("DGEM", dgem_predictions),
            ("Pathway", pathway_predictions),
            ("Proximity", proximity_predictions),
            ("Literature", literature_predictions),
        ]
        for name, preds in summaries:
            if preds:
                print(f"\n  {name} Top {min(5, len(preds))} "
                      f"for {disease_name}:")
                for i, pred in enumerate(preds[:5]):
                    extra = ""
                    if "pubmed_count" in pred:
                        extra = f" [{pred['pubmed_count']} papers]"
                    print(f"    {i+1}. {pred['drug']} "
                          f"(score: {pred['score']:.4f}{extra})")

        return results

    def evaluate(self):
        """Run evaluation on the trained GNN model."""
        if not self.gnn.is_trained:
            raise RuntimeError("Train or load the model first!")

        print("\n[SYSTEM] Running evaluation...")
        return self.evaluator.evaluate_txgnn(self.gnn.tx_model)

    def explain_with_graphmask(self, relation="indication"):
        """Train GraphMask explainability on the current model."""
        if not self.gnn.is_trained:
            raise RuntimeError("Train or load the model first!")

        print("\n[SYSTEM] Training GraphMask explainability...")
        self.explainer.train_graphmask(self.gnn.tx_model, relation)

    def list_diseases(self):
        """List all diseases in the knowledge graph."""
        return self.gnn.get_all_diseases()

    def list_drugs(self):
        """List all drugs in the knowledge graph."""
        return self.gnn.get_all_drugs()


# ─────────────────────────────────────────────────────────────
# Command-line interface
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AI Drug Repurposing System"
    )
    parser.add_argument(
        "--disease", type=str, default="Alzheimer disease",
        help="Disease name to find repurposing candidates for"
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="Number of drug candidates to return"
    )
    parser.add_argument(
        "--explain-top", type=int, default=5,
        help="Number of top drugs to explain in detail"
    )
    parser.add_argument(
        "--split", type=str, default="random",
        choices=["random", "complex_disease", "full_graph"],
        help="Data split type for GNN evaluation"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of fine-tuning epochs"
    )
    parser.add_argument(
        "--train", action="store_true", default=False,
        help="Train a new model (default: load existing)"
    )
    parser.add_argument(
        "--discover", action="store_true", default=False,
        help="Run node mapping discovery and exit"
    )
    parser.add_argument(
        "--no-dgem", action="store_true", default=False,
        help="Disable DGEM gene expression module"
    )
    parser.add_argument(
        "--no-pathway", action="store_true", default=False,
        help="Disable pathway scoring module"
    )
    parser.add_argument(
        "--no-proximity", action="store_true", default=False,
        help="Disable proximity scoring module"
    )
    parser.add_argument(
        "--no-literature", action="store_true", default=False,
        help="Disable literature mining module"
    )

    args = parser.parse_args()

    system = DrugRepurposingSystem(
        enable_dgem=not args.no_dgem,
        enable_pathway=not args.no_pathway,
        enable_proximity=not args.no_proximity,
        enable_literature=not args.no_literature,
    )

    if args.discover:
        system.gnn.load_knowledge_graph(split=args.split)
        system.gnn.discover_node_mappings()
    else:
        # Set up GNN (train or load)
        system.setup_gnn(
            split=args.split,
            train=args.train,
            epochs=args.epochs
        )

        # Run repurposing
        results = system.repurpose(
            args.disease,
            top_k=args.top_k,
            explain_top=args.explain_top
        )
