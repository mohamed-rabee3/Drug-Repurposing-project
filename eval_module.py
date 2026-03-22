"""
Module 4: Evaluation of Drug Repurposing Predictions
=====================================================
Computes standard metrics: AUPRC, Hits@K, MRR, and plots PR curves.
"""

import torch_cuda_ld_path

torch_cuda_ld_path.apply()

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score
)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import os
import json
from txgnn import TxEval


class RepurposingEvaluator:
    """Evaluate drug repurposing model performance."""

    def __init__(self, output_folder="./outputs"):
        self.output_folder = output_folder
        try:
            os.makedirs(output_folder, exist_ok=True)
        except (FileExistsError, OSError):
            pass

    def evaluate_txgnn(self, tx_model, split_type="test_set"):
        """
        Run the built-in TxGNN evaluation.

        Args:
            tx_model: A trained TxGNN model instance.
            split_type: "test_set" uses the held-out test diseases.

        Returns:
            Dict with evaluation metrics.
        """
        print("[EVAL] Running TxGNN disease-centric evaluation...")

        evaluator = TxEval(model=tx_model)
        result = evaluator.eval_disease_centric(
            disease_idxs=split_type,
            show_plot=False,
            verbose=True,
            save_result=True,
            return_raw=False,
            save_name=os.path.join(self.output_folder, "eval_results")
        )

        print("[EVAL] Evaluation complete!")
        return result

    def compute_hits_at_k(self, true_drugs, predicted_drugs,
                           k_values=None):
        """
        Compute Hits@K metric.

        Args:
            true_drugs: Set of correct drug indices.
            predicted_drugs: Ordered list of predicted drug indices (best first).
            k_values: Which K values to compute.

        Returns:
            Dict like {"hits@1": 0.33, "hits@5": 0.67, ...}
        """
        if k_values is None:
            k_values = [1, 5, 10, 20]

        results = {}
        true_set = set(true_drugs)
        for k in k_values:
            top_k = set(predicted_drugs[:k])
            hits = len(top_k.intersection(true_set))
            results[f"hits@{k}"] = hits / min(k, len(true_set)) if true_set else 0.0
        return results

    def compute_mrr(self, true_drugs, predicted_drugs):
        """
        Compute Mean Reciprocal Rank.

        Args:
            true_drugs: Set of correct drug indices.
            predicted_drugs: Ordered list of predicted drug indices.

        Returns:
            Float MRR score.
        """
        true_set = set(true_drugs)
        for rank, drug in enumerate(predicted_drugs, 1):
            if drug in true_set:
                return 1.0 / rank
        return 0.0

    def compute_auprc(self, y_true, y_scores):
        """
        Compute Area Under Precision-Recall Curve.

        Args:
            y_true: Binary array (1 = true drug, 0 = not).
            y_scores: Predicted scores for each drug.

        Returns:
            Float AUPRC score.
        """
        return average_precision_score(y_true, y_scores)

    def compute_auroc(self, y_true, y_scores):
        """
        Compute Area Under ROC Curve.

        Args:
            y_true: Binary array.
            y_scores: Predicted scores.

        Returns:
            Float AUROC score.
        """
        return roc_auc_score(y_true, y_scores)

    def plot_precision_recall(self, y_true, y_scores,
                               title="Precision-Recall Curve"):
        """
        Plot and save a precision-recall curve.

        Args:
            y_true: Binary array (1 = true drug, 0 = not).
            y_scores: Predicted scores for each drug.
            title: Plot title.

        Returns:
            Float AUPRC score.
        """
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        auprc = average_precision_score(y_true, y_scores)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, "b-", linewidth=2,
                 label=f"AUPRC = {auprc:.4f}")
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(self.output_folder, "precision_recall.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[EVAL] Plot saved to {save_path}")

        return auprc

    def full_evaluation(self, y_true, y_scores, predicted_drugs=None,
                         true_drugs=None):
        """
        Run all evaluation metrics at once.

        Args:
            y_true: Binary labels for all drugs.
            y_scores: Predicted scores for all drugs.
            predicted_drugs: Ordered list of predicted drug indices (optional).
            true_drugs: Set of correct drug indices (optional).

        Returns:
            Dict with all metrics.
        """
        results = {
            "auprc": self.compute_auprc(y_true, y_scores),
        }

        try:
            results["auroc"] = self.compute_auroc(y_true, y_scores)
        except ValueError:
            results["auroc"] = None  # Only one class present

        if predicted_drugs is not None and true_drugs is not None:
            results.update(self.compute_hits_at_k(true_drugs, predicted_drugs))
            results["mrr"] = self.compute_mrr(true_drugs, predicted_drugs)

        # Plot PR curve
        self.plot_precision_recall(y_true, y_scores)

        # Save metrics
        metrics_path = os.path.join(self.output_folder, "evaluation_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[EVAL] Metrics saved to {metrics_path}")

        # Print summary
        print("\n[EVAL] === EVALUATION SUMMARY ===")
        for key, val in results.items():
            if val is not None:
                if isinstance(val, float):
                    print(f"    {key}: {val:.4f}")
                else:
                    print(f"    {key}: {val}")

        return results


if __name__ == "__main__":
    print("Evaluation module ready. Use with a trained TxGNN model.")
    print("\nExample with synthetic data:")

    # Demo with synthetic data
    np.random.seed(42)
    n_drugs = 100
    y_true = np.zeros(n_drugs)
    y_true[:10] = 1  # 10 true drugs
    y_scores = np.random.rand(n_drugs) * 0.5
    y_scores[:10] += 0.4  # Boost true drugs

    evaluator = RepurposingEvaluator()
    results = evaluator.full_evaluation(
        y_true, y_scores,
        predicted_drugs=list(np.argsort(y_scores)[::-1]),
        true_drugs=set(range(10))
    )
    print(f"\nDemo AUPRC: {results['auprc']:.4f}")
