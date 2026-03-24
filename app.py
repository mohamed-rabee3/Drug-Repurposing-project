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

# ─── Main Area ───
disease_input = st.text_input(
    "Enter a disease name:",
    placeholder="e.g., Alzheimer disease, type 2 diabetes, breast cancer"
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

        # Build tabs for each module that has results
        tab_names = []
        tab_data = []

        module_configs = [
            ("TxGNN", "txgnn_predictions",
             "Knowledge graph embedding similarity"),
            ("DGEM", "dgem_predictions",
             "Gene expression reversal score"),
            ("Pathway", "pathway_predictions",
             "Pathway enrichment (Fisher's exact test)"),
            ("Proximity", "proximity_predictions",
             "PPI network distance"),
            ("Literature", "literature_predictions",
             "PubMed co-occurrence + LLM relevance"),
        ]

        for name, key, desc in module_configs:
            preds = results.get(key, [])
            if preds:
                tab_names.append(name)
                tab_data.append((preds, desc, key))

        if tab_names:
            tabs = st.tabs(tab_names)
            for tab, (preds, desc, key) in zip(tabs, tab_data):
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
                        color = ("green" if score > 0.7
                                 else "orange" if score > 0.4
                                 else "red")
                        extra = ""
                        if "pubmed_count" in pred:
                            extra = f" ({pred['pubmed_count']} papers)"
                        st.markdown(
                            f"**{i+1}. {pred['drug']}** — "
                            f":{color}[{score:.4f}]{extra}"
                        )
        else:
            st.info("No predictions available.")

        # Explanations
        if results.get("explanations"):
            st.subheader("Biological Explanations")
            for drug, expl in results["explanations"].items():
                with st.expander(f"📋 {drug}"):
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
