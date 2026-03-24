"""
Network Data Setup Script
==========================
Parses TxGNN's kg.csv to extract subgraphs needed by the
Pathway and Proximity scoring modules.

Run once:  python scripts/setup_network.py
"""

import os
import csv
import pickle
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")
NETWORK_FOLDER = os.path.join(DATA_FOLDER, "network")
KG_FILE = os.path.join(DATA_FOLDER, "kg.csv")


def ensure_dirs():
    os.makedirs(NETWORK_FOLDER, exist_ok=True)
    print(f"[SETUP] Output directory: {NETWORK_FOLDER}")


def parse_kg():
    """
    Parse kg.csv and extract the 4 relation types we need.

    kg.csv columns (comma-separated):
      relation, display_relation, x_index, x_id, x_type, x_name,
      x_source, y_index, y_id, y_type, y_name, y_source
    """
    drug_targets = defaultdict(set)       # drug_name_lower -> {protein_name}
    disease_proteins = defaultdict(set)   # disease_name_lower -> {protein_name}
    pathway_proteins = defaultdict(set)   # pathway_name -> {protein_name}
    protein_pathways = defaultdict(set)   # protein_name -> {pathway_name}
    ppi_edges = []                        # [(protein_a, protein_b)]

    target_relations = {
        "drug_protein": "drug_targets",
        "disease_protein": "disease_proteins",
        "pathway_protein": "pathway_proteins",
        "protein_protein": "ppi",
    }

    if not os.path.exists(KG_FILE):
        print(f"[ERROR] kg.csv not found at {KG_FILE}")
        print("[ERROR] Run TxGNN data loading first (python main.py --discover)")
        return None

    print(f"[STEP 1] Parsing {KG_FILE}...")
    print("[STEP 1] This may take 30-60 seconds for ~8.5M rows...")

    counts = defaultdict(int)
    row_count = 0

    with open(KG_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            rel = row["relation"]

            if rel == "drug_protein":
                drug_name = row["x_name"].lower().strip()
                protein_name = row["y_name"].strip()
                drug_targets[drug_name].add(protein_name)
                counts["drug_protein"] += 1

            elif rel == "disease_protein":
                disease_name = row["x_name"].lower().strip()
                protein_name = row["y_name"].strip()
                disease_proteins[disease_name].add(protein_name)
                counts["disease_protein"] += 1

            elif rel == "pathway_protein":
                pathway_name = row["x_name"].strip()
                protein_name = row["y_name"].strip()
                pathway_proteins[pathway_name].add(protein_name)
                protein_pathways[protein_name].add(pathway_name)
                counts["pathway_protein"] += 1

            elif rel == "protein_protein":
                prot_a = row["x_name"].strip()
                prot_b = row["y_name"].strip()
                ppi_edges.append((prot_a, prot_b))
                counts["protein_protein"] += 1

            if row_count % 2_000_000 == 0:
                print(f"  ...processed {row_count:,} rows")

    print(f"[STEP 1] Done. Processed {row_count:,} total rows.")
    for rel, cnt in sorted(counts.items()):
        print(f"  {rel}: {cnt:,} edges")

    return {
        "drug_targets": dict(drug_targets),
        "disease_proteins": dict(disease_proteins),
        "pathway_proteins": dict(pathway_proteins),
        "protein_pathways": dict(protein_pathways),
        "ppi_edges": ppi_edges,
    }


def build_ppi_graph(ppi_edges):
    """Build a networkx Graph from PPI edges."""
    import networkx as nx

    print(f"[STEP 2] Building PPI graph from {len(ppi_edges):,} edges...")
    G = nx.Graph()
    G.add_edges_from(ppi_edges)
    print(f"[STEP 2] PPI graph: {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} edges")
    return G


def save_data(data, ppi_graph):
    """Save all extracted data as pickle files."""
    print("[STEP 3] Saving pickle files...")

    files = {
        "drug_targets.pkl": data["drug_targets"],
        "disease_proteins.pkl": data["disease_proteins"],
        "pathway_proteins.pkl": data["pathway_proteins"],
        "protein_pathways.pkl": data["protein_pathways"],
        "ppi_graph.pkl": ppi_graph,
    }

    for fname, obj in files.items():
        path = os.path.join(NETWORK_FOLDER, fname)
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=4)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  {fname}: {size_mb:.1f} MB")


def coverage_report(data):
    """Print coverage statistics."""
    print("\n" + "=" * 60)
    print("  NETWORK DATA COVERAGE REPORT")
    print("=" * 60)

    dt = data["drug_targets"]
    dp = data["disease_proteins"]
    pp = data["pathway_proteins"]
    prp = data["protein_pathways"]

    print(f"\n  Drugs with targets: {len(dt):,}")
    print(f"  Diseases with associated proteins: {len(dp):,}")
    print(f"  Pathways: {len(pp):,}")
    print(f"  Proteins in pathways: {len(prp):,}")
    print(f"  PPI edges: {len(data['ppi_edges']):,}")

    # Specific checks
    print("\n  --- Specific Checks ---")
    for drug in ["sulindac", "metformin", "aspirin"]:
        targets = dt.get(drug, set())
        print(f"  {drug}: {len(targets)} targets "
              f"{list(targets)[:5] if targets else '(none)'}")

    for disease in ["fragile x syndrome", "alzheimer disease",
                     "type 2 diabetes mellitus"]:
        proteins = dp.get(disease, set())
        print(f"  '{disease}': {len(proteins)} associated proteins")

    print("\n" + "=" * 60)


def main():
    print("=" * 60)
    print("  NETWORK DATA SETUP")
    print("  Pathway + Proximity Scoring Modules")
    print("=" * 60)

    ensure_dirs()

    # Check if already processed
    required = ["drug_targets.pkl", "disease_proteins.pkl",
                "pathway_proteins.pkl", "protein_pathways.pkl",
                "ppi_graph.pkl"]
    all_exist = all(
        os.path.exists(os.path.join(NETWORK_FOLDER, f)) for f in required
    )
    if all_exist:
        print("[SETUP] All network data files already exist.")
        print("[SETUP] Delete data/network/ to re-process.")
        return

    data = parse_kg()
    if data is None:
        return

    ppi_graph = build_ppi_graph(data["ppi_edges"])
    save_data(data, ppi_graph)
    coverage_report(data)

    print("\n[SETUP] Done! Network data ready for pathway and proximity modules.")


if __name__ == "__main__":
    main()
