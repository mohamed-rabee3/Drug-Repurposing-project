"""
Streamlit Web Interface for AI Drug Repurposing
=================================================
Run with: streamlit run app.py
"""

import torch_cuda_ld_path

torch_cuda_ld_path.apply()

import streamlit as st
import json
import os
import pandas as pd

# ─── Page Config ───
st.set_page_config(
    page_title="AI Drug Repurposing",
    page_icon="💊",
    layout="wide"
)


# ─── Cache the heavy system initialization ───
@st.cache_resource
def load_system(split, enable_dgem, enable_pathway,
                enable_proximity, enable_literature):
    """Load and cache the system so it persists across Streamlit reruns."""
    from main import DrugRepurposingSystem
    system = DrugRepurposingSystem(
        enable_dgem=enable_dgem,
        enable_pathway=enable_pathway,
        enable_proximity=enable_proximity,
        enable_literature=enable_literature,
    )
    system.setup_gnn(split=split, train=False)
    return system


st.title("💊 AI Drug Repurposing System")
st.caption("TxGNN + DGEM + Pathway + Proximity + Literature")

# ─── Sidebar ───
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of drug candidates", 5, 50, 20)
    explain_top = st.slider("Drugs to explain in detail", 0, 10, 5)
    split = st.selectbox("Evaluation split", ["random", "complex_disease"])

    st.divider()
    st.header("Scoring Modules")
    enable_dgem = st.checkbox("DGEM (Gene Expression)", value=True)
    enable_pathway = st.checkbox("Pathway Enrichment", value=True)
    enable_proximity = st.checkbox("Network Proximity", value=True)
    enable_literature = st.checkbox("Literature Mining", value=True)

    if st.button("Initialize System"):
        with st.spinner("Loading TxGNN Knowledge Graph... "
                       "(first time takes a few minutes)"):
            st.session_state.system = load_system(
                split, enable_dgem, enable_pathway,
                enable_proximity, enable_literature
            )
        st.success("System ready!")

    st.divider()
    st.markdown("""
    **Setup:**
    - `python scripts/setup_dgem.py` (DGEM data)
    - `python scripts/setup_network.py` (Pathway + Proximity data)
    """)

# ─── Load disease names for autocomplete ───
@st.cache_data
def load_disease_names(data_folder="./data"):
    """Load all disease names from node.csv for autocomplete."""
    nodes_file = os.path.join(data_folder, "node.csv")
    if not os.path.exists(nodes_file):
        return []
    df = pd.read_csv(nodes_file, sep='\t', on_bad_lines='skip')
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip('"')
    diseases = df[df['node_type'] == 'disease']['node_name'].dropna().tolist()
    return sorted(set(diseases))

disease_list = load_disease_names()

# ─── Main Area ───
disease_input = st.selectbox(
    "Enter a disease name:",
    options=[""] + disease_list,
    index=0,
    placeholder="Type to search... e.g., Alzheimer, fragile x, diabetes",
)

if st.button("Find Repurposing Candidates", type="primary"):
    if "system" not in st.session_state:
        st.error("Initialize the system first (sidebar button).")
    elif not disease_input:
        st.warning("Please enter a disease name.")
    else:
        with st.spinner(f"Analyzing {disease_input}..."):
            try:
                results = st.session_state.system.repurpose(
                    disease_input,
                    top_k=top_k,
                    explain_top=explain_top
                )
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        # Display results
        st.header(f"Results for: {disease_input}")

        # Disease context
        if results.get("disease_context"):
            ctx = results["disease_context"]
            with st.expander("Disease Context", expanded=False):
                if isinstance(ctx, dict):
                    st.write(f"**Category:** "
                             f"{ctx.get('disease_category', 'N/A')}")
                    st.write(f"**Pathophysiology:** "
                             f"{ctx.get('pathophysiology_summary', 'N/A')}")
                    if ctx.get("key_pathways"):
                        st.write(f"**Key Pathways:** "
                                 f"{', '.join(ctx['key_pathways'])}")
                    if ctx.get("key_targets"):
                        st.write(f"**Key Targets:** "
                                 f"{', '.join(ctx['key_targets'])}")

        # ── TxGNN & DGEM side by side ──
        txgnn_preds = results.get("txgnn_predictions", [])
        dgem_preds = results.get("dgem_predictions", [])

        if txgnn_preds or dgem_preds:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("TxGNN Predictions")
                st.caption("Knowledge graph embedding similarity")
                meta = results.get("metadata", {})
                if txgnn_preds:
                    for i, pred in enumerate(txgnn_preds):
                        score = pred["score"]
                        pct = score * 100
                        color = ("green" if score > 0.7
                                 else "orange" if score > 0.4
                                 else "red")
                        st.markdown(
                            f"**{i+1}. {pred['drug']}** — "
                            f":{color}[{pct:.0f}%]"
                        )
                else:
                    st.info("No TxGNN predictions.")

            with col2:
                st.subheader("DGEM Predictions")
                st.caption("Gene expression reversal score")
                scored = meta.get("dgem_drugs_scored", 0)
                if scored:
                    st.info(f"Scored {scored} drugs")
                if dgem_preds:
                    for i, pred in enumerate(dgem_preds):
                        score = pred["score"]
                        pct = score * 100
                        color = ("green" if score > 0.7
                                 else "orange" if score > 0.4
                                 else "red")
                        st.markdown(
                            f"**{i+1}. {pred['drug']}** — "
                            f":{color}[{pct:.0f}%]"
                        )
                else:
                    st.info("No DGEM predictions.")
        else:
            st.info("No predictions available.")

        # ── Other modules in tabs ──
        other_modules = [
            ("Pathway", "pathway_predictions",
             "Pathway enrichment (Fisher's exact test)"),
            ("Proximity", "proximity_predictions",
             "PPI network distance"),
            ("Literature", "literature_predictions",
             "PubMed co-occurrence + LLM relevance"),
        ]

        other_tab_names = []
        other_tab_data = []
        for name, key, desc in other_modules:
            preds = results.get(key, [])
            if preds:
                other_tab_names.append(name)
                other_tab_data.append((preds, desc, key))

        if other_tab_names:
            st.divider()
            st.subheader("Other Scoring Modules")
            tabs = st.tabs(other_tab_names)
            for tab, (preds, desc, key) in zip(tabs, other_tab_data):
                with tab:
                    st.caption(desc)
                    meta = results.get("metadata", {})
                    scored_key = key.replace("_predictions",
                                            "_drugs_scored")
                    scored = meta.get(scored_key, 0)
                    if scored:
                        st.info(f"Scored {scored} drugs")

                    for i, pred in enumerate(preds):
                        score = pred["score"]
                        pct = score * 100
                        color = ("green" if score > 0.7
                                 else "orange" if score > 0.4
                                 else "red")
                        extra = ""
                        if "pubmed_count" in pred:
                            extra = f" ({pred['pubmed_count']} papers)"
                        st.markdown(
                            f"**{i+1}. {pred['drug']}** — "
                            f":{color}[{pct:.0f}%]{extra}"
                        )

        # Explanations
        if results.get("explanations"):
            st.divider()
            st.subheader("Biological Explanations (TxGNN + DGEM Top Candidates)")
            # Build source labels
            txgnn_drugs = {p["drug"].lower() for p in txgnn_preds[:5]}
            dgem_drugs_set = {p["drug"].lower() for p in dgem_preds[:5]}
            for drug, expl in results["explanations"].items():
                sources = []
                if drug.lower() in txgnn_drugs:
                    sources.append("TxGNN")
                if drug.lower() in dgem_drugs_set:
                    sources.append("DGEM")
                label = f" [{', '.join(sources)}]" if sources else ""
                with st.expander(f"📋 {drug}{label}"):
                    if isinstance(expl, dict):
                        st.write(f"**Summary:** "
                                 f"{expl.get('summary', 'N/A')}")
                        st.write(f"**Mechanism:** "
                                 f"{expl.get('mechanism_of_action', 'N/A')}")
                        st.write(f"**Pathway:** "
                                 f"{expl.get('biological_pathway', 'N/A')}")
                        if expl.get("target_proteins"):
                            st.write(f"**Target Proteins:** "
                                     f"{', '.join(expl['target_proteins'])}")
                    else:
                        st.write(str(expl))

        # Full report
        if results.get("report"):
            st.subheader("Full Report")
            st.markdown(results["report"])

        # Metadata
        if results.get("metadata"):
            with st.expander("Metadata"):
                st.json(results["metadata"])

        # Download
        st.download_button(
            "Download Full Results (JSON)",
            json.dumps(results, indent=2, default=str),
            file_name=f"repurposing_{disease_input.replace(' ', '_')}.json",
            mime="application/json"
        )
