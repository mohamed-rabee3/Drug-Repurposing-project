"""
Module 3: Explainability (GraphMask + NLP)
==========================================
Combines TxGNN's GraphMask XAI with GPT-OSS 20B explanations
to provide multi-hop biological pathway rationales.
"""

import json
import os
import pickle


class RepurposingExplainer:
    """Generate explainable rationales for drug-disease predictions."""

    def __init__(self, nlp_module, output_folder="./outputs"):
        """
        Args:
            nlp_module: An initialized BiomedicalNLP instance.
            output_folder: Where to save explanation outputs.
        """
        self.nlp = nlp_module
        self.output_folder = output_folder
        try:
            os.makedirs(output_folder, exist_ok=True)
        except (FileExistsError, OSError):
            pass

    def train_graphmask(self, tx_model, relation="indication"):
        """
        Train the GraphMask XAI model on top of a trained TxGNN model.

        Identifies which EDGES in the knowledge graph are most important
        for each prediction.

        Args:
            tx_model: A trained TxGNN model instance.
            relation: "indication" or "contraindication".
        """
        print(f"[XAI] Training GraphMask for '{relation}'...")
        print("[XAI] This may take 10-30 minutes on GTX 1650...")

        tx_model.train_graphmask(
            relation=relation,
            learning_rate=3e-4,
            allowance=0.005,
            epochs_per_layer=3,
            penalty_scaling=1,
            valid_per_n=20
        )

        save_path = os.path.join(self.output_folder, f"graphmask_{relation}.pkl")
        print(f"[XAI] GraphMask trained! Results at: {save_path}")

    def get_graph_explanation(self, tx_model, drug_idx, disease_idx,
                              relation="indication"):
        """
        Get the multi-hop path explanation from GraphMask.

        Returns the important edges connecting a drug to a disease
        through intermediate nodes (proteins, pathways, etc.)

        Args:
            tx_model: Trained TxGNN with GraphMask.
            drug_idx: Drug node index.
            disease_idx: Disease node index.
            relation: "indication" or "contraindication".

        Returns:
            List of important edges forming the explanation path,
            or None if GraphMask hasn't been trained.
        """
        gates_path = os.path.join(
            self.output_folder, f"graphmask_output_{relation}.pkl"
        )
        if os.path.exists(gates_path):
            with open(gates_path, "rb") as f:
                gates = pickle.load(f)
            return gates
        else:
            print("[XAI] GraphMask not trained yet. "
                  "Train with train_graphmask() first.")
            return None

    def explain_prediction(self, drug_name, disease_name, gnn_score,
                            graph_paths=None, dgem_score=None):
        """
        Generate a complete explanation combining graph paths + NLP + DGEM.

        Args:
            drug_name: Name of the predicted drug.
            disease_name: Name of the target disease.
            gnn_score: The GNN's confidence score (0-1).
            graph_paths: Optional multi-hop paths from GraphMask.
            dgem_score: Optional DGEM gene expression reversal score (0-1).

        Returns:
            Dict with combined explanation.
        """
        # Get NLP explanation
        nlp_explanation = self.nlp.explain_drug_disease_relationship(
            drug_name, disease_name
        )

        explanation = {
            "drug": drug_name,
            "disease": disease_name,
            "gnn_confidence_score": gnn_score,
            "confidence_level": self._score_to_confidence(gnn_score),
            "graph_paths": graph_paths,
            "biological_explanation": nlp_explanation,
        }

        # Add DGEM evidence if available
        if dgem_score is not None:
            explanation["dgem_score"] = dgem_score
            explanation["dgem_interpretation"] = self._dgem_to_interpretation(dgem_score)

        return explanation

    def explain_batch(self, disease_name, predictions, max_explain=5):
        """
        Generate explanations for a batch of drug predictions.

        Args:
            disease_name: Target disease.
            predictions: List of prediction dicts from the GNN module.
            max_explain: Maximum number of drugs to explain.

        Returns:
            List of explanation dicts.
        """
        explanations = []
        for pred in predictions[:max_explain]:
            drug_name = pred.get("drug", pred.get("drug_name", f"drug_{pred['drug_idx']}"))
            print(f"[XAI] Explaining: {drug_name} -> {disease_name}...")

            expl = self.explain_prediction(
                drug_name=drug_name,
                disease_name=disease_name,
                gnn_score=pred["score"],
                dgem_score=pred.get("dgem_score"),
            )
            explanations.append(expl)

        return explanations

    def save_explanations(self, explanations, filename="explanations.json"):
        """Save explanations to a JSON file."""
        output_path = os.path.join(self.output_folder, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(explanations, f, indent=2, default=str)
        print(f"[XAI] Explanations saved to {output_path}")
        return output_path

    @staticmethod
    def _score_to_confidence(score):
        """Convert a numeric score to a human-readable confidence level."""
        if score >= 0.9:
            return "Very High - Strong candidate for further investigation"
        elif score >= 0.7:
            return "High - Promising candidate"
        elif score >= 0.5:
            return "Moderate - Worth exploring"
        elif score >= 0.3:
            return "Low - Weak signal, needs more evidence"
        else:
            return "Very Low - Unlikely candidate"

    @staticmethod
    def _dgem_to_interpretation(dgem_score):
        """Convert DGEM reversal score to human-readable interpretation."""
        if dgem_score >= 0.75:
            return ("Strong reversal - Drug significantly reverses the disease's "
                    "gene expression signature")
        elif dgem_score >= 0.60:
            return ("Moderate reversal - Drug partially reverses disease "
                    "expression patterns")
        elif dgem_score >= 0.50:
            return ("Weak reversal - Slight tendency to reverse disease "
                    "expression, near neutral")
        elif dgem_score >= 0.40:
            return ("Neutral - No clear reversal or similarity in expression")
        else:
            return ("Same direction - Drug expression aligns with disease "
                    "rather than reversing it")


if __name__ == "__main__":
    print("Explainability module ready.")
    print("Use with a trained TxGNN model and BiomedicalNLP instance.")
    print("Example:")
    print("  from nlp_module import BiomedicalNLP")
    print("  from explain_module import RepurposingExplainer")
    print("  nlp = BiomedicalNLP()")
    print("  explainer = RepurposingExplainer(nlp)")
    print("  expl = explainer.explain_prediction('Metformin', 'Alzheimer', 0.85)")
