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
def load_system(split):
    """Load and cache the system so it persists across Streamlit reruns."""
    from main import DrugRepurposingSystem
    system = DrugRepurposingSystem()
    system.setup_gnn(split=split, train=False)
    return system


st.title("💊 AI Drug Repurposing System")
st.caption("TxGNN + GPT-OSS 20B — Predict new uses for existing drugs")

# ─── Sidebar ───
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of drug candidates", 5, 50, 20)
    explain_top = st.slider("Drugs to explain in detail", 1, 10, 5)
    split = st.selectbox("Evaluation split", ["random", "complex_disease"])

    if st.button("Initialize System"):
        with st.spinner("Loading TxGNN Knowledge Graph... "
                       "(first time takes a few minutes)"):
            st.session_state.system = load_system(split)
        st.success("System ready!")

    st.divider()
    st.markdown("""
    **How to use:**
    1. Click **Initialize System** (first time loads ~1.5GB KG data)
    2. Enter a disease name
    3. Click **Find Repurposing Candidates**

    **Note:** The system must be trained before use.
    See README.md for training instructions.
    """)

# ─── Main Area ───
disease_input = st.text_input(
    "Enter a disease name:",
    placeholder="e.g., Alzheimer disease, type 2 diabetes, breast cancer"
)

if st.button("Find Repurposing Candidates", type="primary"):
    if "system" not in st.session_state:
        st.error("Initialize the system first (click the button in the sidebar).")
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
                    if ctx.get("current_treatments"):
                        st.write(f"**Current Treatments:** "
                                 f"{', '.join(ctx['current_treatments'])}")
                    if ctx.get("unmet_needs"):
                        st.write(f"**Unmet Needs:** {ctx['unmet_needs']}")

        # Predictions table
        st.subheader("Drug Candidates (ranked by GNN score)")
        if results.get("predictions"):
            for i, pred in enumerate(results["predictions"]):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{i+1}. {pred['drug']}**")
                with col2:
                    score = pred["score"]
                    if score > 0.7:
                        color = "green"
                    elif score > 0.4:
                        color = "orange"
                    else:
                        color = "red"
                    st.markdown(f":{color}[Score: {score:.4f}]")
        else:
            st.info("No GNN predictions available. "
                    "Ensure the model is trained and disease name mappings "
                    "are configured.")

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
                        st.write(f"**Evidence:** "
                                 f"{expl.get('evidence_strength', 'N/A')}")
                        if expl.get("target_proteins"):
                            st.write(f"**Target Proteins:** "
                                     f"{', '.join(expl['target_proteins'])}")
                        if expl.get("safety_considerations"):
                            st.write(f"**Safety:** "
                                     f"{expl['safety_considerations']}")
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
