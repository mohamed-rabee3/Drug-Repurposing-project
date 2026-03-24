"""
Module 1: TxGNN Graph Neural Network for Drug Repurposing
==========================================================
This module handles:
  - Loading the pre-built TxGNN knowledge graph
  - Training (or loading) the GNN model
  - Predicting drug candidates for a given disease
"""

import torch_cuda_ld_path

torch_cuda_ld_path.apply()

import txgnn_pandas_patch

txgnn_pandas_patch.apply()

import os
import sys
import torch
import pickle
import json
import numpy as np
import pandas as pd
from txgnn import TxData, TxGNN, TxEval

import dgl_dataloader_compat

dgl_dataloader_compat.apply()


class DrugRepurposingGNN:
    """Wrapper around TxGNN for easy drug repurposing predictions."""

    def __init__(self, data_folder="./data", model_folder="./models"):
        """
        Initialize the GNN module.

        Args:
            data_folder: Where to store/load the knowledge graph data.
                         TxGNN will auto-download it here (~1.5 GB).
            model_folder: Where to save/load trained model checkpoints.
        """
        self.data_folder = data_folder
        self.model_folder = model_folder
        # DGL Windows CUDA wheels do not implement COOToCSR / in_degrees on GPU for
        # heterogeneous graphs; TxGNN hits that in model_initialize (Full_Graph_NegSampler).
        # Linux/WSL2 can use CUDA when available.
        cuda_ok = torch.cuda.is_available()
        if cuda_ok and sys.platform != "win32":
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        self.tx_data = None
        self.tx_model = None
        self.tx_eval = None
        self.is_trained = False
        self._disease_names = {}  # idx -> name mapping
        self._drug_ids = {}       # idx -> DrugBank ID mapping (for DGEM)
        self._drug_names = {}     # idx -> name mapping

        for d in (data_folder, model_folder):
            try:
                os.makedirs(d, exist_ok=True)
            except (FileExistsError, OSError):
                pass

        print(f"[GNN] Device: {self.device}")
        if cuda_ok:
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[GNN] CUDA GPU: {gpu_name} ({vram:.1f} GB VRAM)")
            if sys.platform == "win32":
                print(
                    "[GNN] Windows: TxGNN+DGL run on CPU (DGL heterograph CUDA ops unsupported on Windows)."
                )
                print(
                    "[GNN] For GPU-accelerated training use WSL2 (Ubuntu) or native Linux with CUDA."
                )

    def load_knowledge_graph(self, split="random", seed=42):
        """
        Load the TxGNN pre-built knowledge graph.

        First time: Downloads ~1.5 GB from Harvard Dataverse.
        Subsequent times: Loads from local cache.

        Args:
            split: How to split data for evaluation.
                   "random"          - random 90/5/5 split (recommended for beginners)
                   "complex_disease" - holds out entire disease areas (requires PyG)
                   "full_graph"      - use everything for training
            seed: Random seed for reproducibility.
        """
        if split == "complex_disease":
            try:
                import torch_geometric
            except ImportError:
                print("[GNN] WARNING: 'complex_disease' split requires PyG.")
                print("[GNN]   Install: pip install torch-geometric torch-sparse torch-scatter")
                print("[GNN]   Or use split='random' instead.")
                raise ImportError("PyG required for complex_disease split.")

        print(f"[GNN] Loading knowledge graph (split={split})...")
        print(f"[GNN] Data folder: {self.data_folder}")
        print("[GNN] First run will download ~1.5 GB — this may take a few minutes...")

        self.tx_data = TxData(data_folder_path=self.data_folder)
        self.tx_data.prepare_split(split=split, seed=seed)

        print("[GNN] Knowledge graph loaded successfully!")
        print(f"[GNN] Split: {split} | Seed: {seed}")

        # Load node name mappings
        self._load_node_names()
        if not self._disease_names:
            print("[GNN] Node names not loaded — run discover_node_mappings()")

    def initialize_model(self, n_hid=100, n_inp=100, n_out=100,
                         proto=True, proto_num=3):
        """
        Initialize a fresh TxGNN model.

        Args:
            n_hid: Hidden layer dimensions (100 is the paper's default).
            n_inp: Input dimensions.
            n_out: Output dimensions.
            proto: Use metric learning for zero-shot? (True recommended).
            proto_num: How many similar diseases to use for augmentation.
        """
        if self.tx_data is None:
            raise RuntimeError("Load knowledge graph first! Call load_knowledge_graph()")

        print(f"[GNN] Initializing model (hidden={n_hid}, proto={proto})...")

        self.tx_model = TxGNN(
            data=self.tx_data,
            weight_bias_track=False,
            proj_name="DrugRepurpose",
            exp_name="experiment_1",
            device=self.device
        )

        self.tx_model.model_initialize(
            n_hid=n_hid,
            n_inp=n_inp,
            n_out=n_out,
            proto=proto,
            proto_num=proto_num,
            attention=False,
            sim_measure="all_nodes_profile",
            agg_measure="rarity",
            num_walks=200,
            walk_mode="bit",
            path_length=2,
        )

        print("[GNN] Model initialized!")

    def train(self, pretrain_epochs=2, finetune_epochs=500,
              pretrain_lr=1e-3, finetune_lr=5e-4,
              batch_size=1024):
        """
        Train the GNN model in two phases.

        Phase 1 — Pre-training: Learns general graph structure.
        Phase 2 — Fine-tuning: Focuses on drug-disease predictions.

        Args:
            pretrain_epochs: Number of pre-training epochs (2 is default).
            finetune_epochs: Number of fine-tuning epochs (500 for full, 50 for quick test).
            pretrain_lr: Pre-training learning rate.
            finetune_lr: Fine-tuning learning rate.
            batch_size: Batch size. 1024 works on 4 GB VRAM. Reduce if OOM.
        """
        if self.tx_model is None:
            raise RuntimeError("Initialize model first! Call initialize_model()")

        # Phase 1: Pre-training
        print(f"\n[GNN] === PHASE 1: Pre-training ({pretrain_epochs} epochs) ===")
        print(f"[GNN] Learning rate: {pretrain_lr} | Batch size: {batch_size}")

        self.tx_model.pretrain(
            n_epoch=pretrain_epochs,
            learning_rate=pretrain_lr,
            batch_size=batch_size,
            train_print_per_n=5
        )

        print("[GNN] Pre-training complete!")

        # Phase 2: Fine-tuning
        finetune_path = os.path.join(self.model_folder, "finetune_result")
        print(f"\n[GNN] === PHASE 2: Fine-tuning ({finetune_epochs} epochs) ===")
        print(f"[GNN] Learning rate: {finetune_lr}")

        self.tx_model.finetune(
            n_epoch=finetune_epochs,
            learning_rate=finetune_lr,
            train_print_per_n=5,
            valid_per_n=20,
            save_name=finetune_path + "_metrics"
        )

        # Save the actual model weights (finetune only saves eval metrics)
        model_path = os.path.join(self.model_folder, "finetune_result")
        self.tx_model.save_model(model_path)

        self.is_trained = True
        print(f"\n[GNN] Training complete! Model saved to: {model_path}")

    def save_model(self, path=None):
        """Save the trained model checkpoint."""
        if path is None:
            path = os.path.join(self.model_folder, "model_ckpt")
        print(f"[GNN] Model was auto-saved by finetune() to: "
              f"{os.path.join(self.model_folder, 'finetune_result')}")

    def load_model(self, path=None):
        """Load a previously trained model checkpoint."""
        if path is None:
            path = os.path.join(self.model_folder, "finetune_result")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No model found at {path}. Train first!\n"
                f"Contents of model folder: {os.listdir(self.model_folder)}"
            )
        self.tx_model.load_pretrained(path)
        self.is_trained = True
        print(f"[GNN] Model loaded from {path}")

    def predict_drugs_for_disease(self, disease_idx, top_k=20,
                                   relation="indication"):
        """
        Predict which drugs could treat a given disease.

        Args:
            disease_idx: Numeric disease index in the KG.
            top_k: How many top drug candidates to return.
            relation: "indication" (what treats) or "contraindication" (what harms).

        Returns:
            List of dicts: [{"drug_idx": 42, "score": 0.85, "drug_name": "..."}, ...]
        """
        if not self.is_trained:
            raise RuntimeError("Train or load the model first!")

        print(f"[GNN] Predicting drugs for disease index {disease_idx} "
              f"(top {top_k}, relation={relation})...")

        predictions = self._score_all_drugs(disease_idx, relation, top_k)
        print(f"[GNN] Found {len(predictions)} candidates")
        return predictions

    def _score_all_drugs(self, disease_idx, relation, top_k):
        """
        Score all drugs against a single disease using the trained model.

        Uses TxGNN's retrieve_embedding() to get node embeddings, then
        scores all drugs against the target disease via dot-product similarity.
        """
        try:
            # retrieve_embedding() calls model(G, G, return_h=True) internally,
            # which returns a dict of {node_type: embedding_tensor}
            h = self.tx_model.retrieve_embedding()

            disease_emb = h['disease'][disease_idx]
            drug_embs = h['drug']

            # Score via dot product + sigmoid
            scores = torch.sigmoid(torch.matmul(drug_embs, disease_emb))
            scores = scores.cpu().numpy()

            drug_names = self._drug_names

            # Rank and return top-K
            ranked_indices = np.argsort(scores)[::-1][:top_k]
            results = []
            for idx in ranked_indices:
                name = drug_names.get(int(idx), f"drug_{idx}") if drug_names else f"drug_{idx}"
                results.append({
                    "drug_idx": int(idx),
                    "score": float(scores[idx]),
                    "drug_name": name,
                    "drug": name,
                })

            return results

        except Exception as e:
            print(f"[GNN] WARNING: Direct scoring failed: {e}")
            print("[GNN] Falling back to empty list. NLP module will handle naming.")
            return []

    def discover_node_mappings(self):
        """
        Discovery method — run after loading the KG to find out what
        attributes TxData has and how to map node indices to names.
        """
        print("\n" + "=" * 60)
        print("  KNOWLEDGE GRAPH DISCOVERY")
        print("=" * 60)

        # 1. TxData attributes
        print("\n[1] TxData attributes:")
        for attr in sorted(dir(self.tx_data)):
            if not attr.startswith('_'):
                val = getattr(self.tx_data, attr)
                if not callable(val):
                    print(f"    {attr}: {type(val).__name__}", end="")
                    if hasattr(val, '__len__'):
                        print(f" (len={len(val)})", end="")
                    print()

        # 2. Files in data folder
        print(f"\n[2] Files in data folder ({self.data_folder}):")
        for root, dirs, files in os.walk(self.data_folder):
            for f in sorted(files):
                fpath = os.path.join(root, f)
                size = os.path.getsize(fpath)
                rel = os.path.relpath(fpath, self.data_folder)
                print(f"    {rel} ({size / 1e6:.1f} MB)")

        # 3. DGL heterograph info
        if hasattr(self.tx_data, 'G'):
            G = self.tx_data.G
            print(f"\n[3] DGL Heterograph info:")
            print(f"    Node types: {G.ntypes}")
            print(f"    Edge types: {G.etypes}")
            for ntype in G.ntypes:
                print(f"    '{ntype}' nodes: {G.number_of_nodes(ntype)}")
        else:
            print("\n[3] No 'G' attribute found on TxData.")

        # 4. Look for name mapping files
        print(f"\n[4] Looking for name mapping files...")
        import glob as glob_mod
        for ext in ['*.csv', '*.tsv', '*.pkl', '*.pickle', '*.json']:
            files = glob_mod.glob(os.path.join(self.data_folder, '**', ext), recursive=True)
            for f in files:
                print(f"    Found: {f}")

        print("\n" + "=" * 60)
        print("  Use this output to build your name mappings!")
        print("  Edit _load_node_names() based on what you find above.")
        print("=" * 60)

    def _load_node_names(self):
        """
        Load human-readable names for disease and drug nodes from node.csv.

        The TxGNN data folder contains node.csv (tab-separated, with quoted values)
        with columns: node_index, node_id, node_type, node_name, node_source.

        We need to map from the DGL graph's per-type index to node names.
        The node.csv node_index is a global index, but DGL uses per-type indices
        (e.g., disease 0, disease 1, ...). We build the mapping by ordering
        nodes of each type by their global node_index.
        """
        self._disease_names = {}
        self._drug_names = {}
        self._drug_ids = {}

        try:
            nodes_file = os.path.join(self.data_folder, "node.csv")
            if not os.path.exists(nodes_file):
                print("[GNN] WARNING: node.csv not found in data folder.")
                return

            df = pd.read_csv(nodes_file, sep='\t', on_bad_lines='skip')

            # Strip quotes from string columns
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].str.strip('"')

            # Extract diseases: sort by node_index so position = DGL per-type idx
            diseases = df[df['node_type'] == 'disease'].sort_values('node_index')
            for per_type_idx, (_, row) in enumerate(diseases.iterrows()):
                self._disease_names[per_type_idx] = row['node_name']

            # Extract drugs: sort by node_index so position = DGL per-type idx
            drugs = df[df['node_type'] == 'drug'].sort_values('node_index')
            for per_type_idx, (_, row) in enumerate(drugs.iterrows()):
                self._drug_names[per_type_idx] = row['node_name']
                self._drug_ids[per_type_idx] = row['node_id']

            print(f"[GNN] Loaded {len(self._disease_names)} disease names, "
                  f"{len(self._drug_names)} drug names, "
                  f"{len(self._drug_ids)} drug IDs")

        except Exception as e:
            print(f"[GNN] WARNING: Failed to load node names: {e}")
            print("[GNN] Predictions will use numeric indices instead of names.")

    def find_disease_idx(self, disease_name):
        """
        Look up a disease index from its name in the knowledge graph.
        """
        if not self._disease_names:
            raise RuntimeError(
                "Disease name mapping not loaded. Options:\n"
                "  1. Run discover_node_mappings() to inspect data files\n"
                "  2. Customize _load_node_names() with your file paths\n"
                "  3. Use disease_idx directly (numeric index)"
            )

        disease_name_lower = disease_name.lower().strip()
        exact_matches = []
        substring_matches = []

        for idx, name in self._disease_names.items():
            name_lower = name.lower()
            if name_lower == disease_name_lower:
                exact_matches.append((idx, name))
            elif disease_name_lower in name_lower:
                substring_matches.append((idx, name))

        # Prefer exact matches
        if exact_matches:
            print(f"[GNN] Found exact match: '{exact_matches[0][1]}' "
                  f"(idx={exact_matches[0][0]})")
            return exact_matches[0][0]

        if not substring_matches:
            raise ValueError(
                f"Disease '{disease_name}' not found. "
                f"Sample diseases: {list(self._disease_names.values())[:10]}"
            )

        if len(substring_matches) == 1:
            print(f"[GNN] Found: '{substring_matches[0][1]}' "
                  f"(idx={substring_matches[0][0]})")
            return substring_matches[0][0]

        # Multiple matches: prefer shortest name (most specific match)
        substring_matches.sort(key=lambda x: len(x[1]))
        print(f"[GNN] Multiple matches for '{disease_name}':")
        for idx, name in substring_matches[:10]:
            print(f"    idx={idx}: {name}")
        print(f"[GNN] Using best match: {substring_matches[0][1]}")
        return substring_matches[0][0]

    def get_all_diseases(self):
        """Return a list of all disease names in the knowledge graph."""
        if self._disease_names:
            return list(self._disease_names.values())
        print("[GNN] No disease names loaded. Run discover_node_mappings() first.")
        return []

    def get_all_drugs(self):
        """Return a list of all drug names in the knowledge graph."""
        if self._drug_names:
            return list(self._drug_names.values())
        print("[GNN] No drug names loaded. Run discover_node_mappings() first.")
        return []

    def get_drug_id(self, drug_idx):
        """
        Get the DrugBank ID for a drug by its per-type index.

        Args:
            drug_idx: Per-type drug index in the DGL graph.

        Returns:
            DrugBank ID string (e.g., "DB00605") or None.
        """
        return self._drug_ids.get(drug_idx)


# ─────────────────────────────────────────────────────────────
# Quick test (run this file directly to test)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing GNN Module...")
    gnn = DrugRepurposingGNN()

    # Step 1: Load knowledge graph
    gnn.load_knowledge_graph(split="random")

    # Step 2: Discover what's in the data
    gnn.discover_node_mappings()

    # Step 3: Initialize model
    gnn.initialize_model(n_hid=100)

    # Step 4: Train (reduce epochs for quick test)
    gnn.train(pretrain_epochs=1, finetune_epochs=10)

    # Step 5: Check available diseases/drugs
    diseases = gnn.get_all_diseases()
    drugs = gnn.get_all_drugs()
    print(f"\nKG contains {len(diseases)} disease names loaded, "
          f"{len(drugs)} drug names loaded")
    if diseases:
        print(f"Sample diseases: {diseases[:5]}")
    if drugs:
        print(f"Sample drugs: {drugs[:5]}")

    print("\nGNN Module test complete!")
