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
                 output_folder="./outputs", enable_dgem=True):
        """
        Initialize all modules.

        Args:
            data_folder: Where TxGNN knowledge graph data is stored.
            model_folder: Where trained model checkpoints are saved.
            output_folder: Where results are written.
            enable_dgem: If True, initialize DGEM gene expression module.
        """
        self.output_folder = output_folder
        try:
            os.makedirs(output_folder, exist_ok=True)
        except (FileExistsError, OSError):
            pass

        print("=" * 60)
        print("  AI DRUG REPURPOSING SYSTEM")
        print("  TxGNN + DGEM + GPT-OSS 20B (Groq)")
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
                print("\n[SYSTEM] DGEM data not found. Run: python scripts/setup_dgem.py")
                print("[SYSTEM] Continuing without DGEM (GNN-only mode).")

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

        Produces two separate ranked lists:
          - TxGNN top-k: ranked by GNN embedding similarity
          - DGEM top-k: ranked by gene expression reversal score

        Args:
            disease_name: e.g., "Alzheimer disease"
            top_k: Number of drug candidates from each method.
            explain_top: Number of top drugs to explain via NLP.

        Returns:
            Dict with complete results.
        """
        use_dgem = self.dgem is not None and self.dgem._initialized
        n_steps = 5 if use_dgem else 4

        print(f"\n{'='*60}")
        print(f"  REPURPOSING DRUGS FOR: {disease_name}")
        if use_dgem:
            print(f"  Mode: TxGNN + DGEM (separate rankings)")
        else:
            print(f"  Mode: TxGNN only")
        print(f"{'='*60}")
        start_time = time.time()

        # Step 1: Get disease context from NLP
        print(f"\n[Step 1/{n_steps}] Getting disease context from GPT-OSS 20B...")
        disease_context = self.nlp.enrich_disease_context(disease_name)

        # Step 2: Get drug predictions from GNN
        print(f"\n[Step 2/{n_steps}] Predicting top {top_k} drug candidates "
              "with TxGNN...")
        try:
            disease_idx = self.gnn.find_disease_idx(disease_name)
            gnn_predictions = self.gnn.predict_drugs_for_disease(
                disease_idx=disease_idx,
                top_k=top_k
            )
        except (RuntimeError, ValueError) as e:
            print(f"[SYSTEM] GNN prediction failed: {e}")
            print("[SYSTEM] Falling back to NLP-only mode (no GNN scores).")
            gnn_predictions = []

        # Step 3: DGEM gene expression scoring — independent top-k
        dgem_predictions = []
        if use_dgem:
            print(f"\n[Step 3/{n_steps}] Scoring ALL drugs with DGEM "
                  "(gene expression reversal)...")

            # Score all TxGNN drugs (not just GNN top-k)
            all_drug_names = list(
                self.dgem.drug_id_mapping.get("name_to_id", {}).keys()
            ) if self.dgem.drug_id_mapping else []

            dgem_scores = self.dgem.score_drugs_for_disease(
                disease_name, drug_names=all_drug_names
            )

            if dgem_scores:
                # Sort by reversal score descending, take top_k
                sorted_dgem = sorted(
                    dgem_scores.items(), key=lambda x: x[1], reverse=True
                )
                for rank, (drug, score) in enumerate(sorted_dgem[:top_k]):
                    dgem_predictions.append({
                        "drug": drug,
                        "score": round(score, 6),
                        "rank": rank + 1,
                    })
                print(f"[DGEM] Scored {len(dgem_scores)} drugs, "
                      f"returning top {len(dgem_predictions)}")
            else:
                print("[DGEM] No expression data available for this disease.")

        # Step 4: Explain top predictions with NLP
        step_explain = 4 if use_dgem else 3
        print(f"\n[Step {step_explain}/{n_steps}] Explaining top {explain_top} "
              "candidates with GPT-OSS 20B...")
        # Explain GNN top drugs
        drug_names = [p["drug"] for p in gnn_predictions[:explain_top]]
        explanations = (
            self.nlp.batch_explain(disease_name, drug_names)
            if drug_names else {}
        )

        # Step 5: Generate full report
        step_report = 5 if use_dgem else 4
        print(f"\n[Step {step_report}/{n_steps}] Generating comprehensive report...")
        if gnn_predictions:
            report = self.nlp.generate_report(disease_name, gnn_predictions)
        else:
            report = self.nlp.generate_report(
                disease_name, [{"drug": "N/A", "score": 0}]
            )

        # Combine results
        elapsed = time.time() - start_time
        model_name = "TxGNN + DGEM + GPT-OSS-20B" if use_dgem else "TxGNN + GPT-OSS-20B"
        results = {
            "disease": disease_name,
            "disease_context": disease_context,
            "txgnn_predictions": gnn_predictions,
            "explanations": explanations,
            "report": report,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": model_name,
                "top_k": top_k,
                "processing_time_seconds": round(elapsed, 2),
                "dgem_enabled": use_dgem,
            }
        }

        # Add DGEM results as separate section
        if use_dgem:
            results["dgem_predictions"] = dgem_predictions
            results["metadata"]["dgem_drugs_scored"] = len(dgem_scores) if dgem_scores else 0

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

        # Print TxGNN summary
        print(f"\n  TxGNN Top {min(5, len(gnn_predictions))} "
              f"for {disease_name}:")
        for i, pred in enumerate(gnn_predictions[:5]):
            print(f"    {i+1}. {pred['drug']} (score: {pred['score']:.4f})")

        # Print DGEM summary
        if use_dgem and dgem_predictions:
            print(f"\n  DGEM Top {min(5, len(dgem_predictions))} "
                  f"for {disease_name}:")
            for i, pred in enumerate(dgem_predictions[:5]):
                print(f"    {i+1}. {pred['drug']} "
                      f"(reversal: {pred['score']:.4f})")

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

    args = parser.parse_args()

    system = DrugRepurposingSystem(
        enable_dgem=not args.no_dgem,
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
