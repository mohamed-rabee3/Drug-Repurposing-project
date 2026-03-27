"""
Streamlit Web Interface for AI Drug Repurposing
=================================================
Run with: streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
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
@st.cache_resource(show_spinner=False)
def load_system(enable_dgem, enable_pathway,
                enable_proximity, enable_literature):
    """Load and cache the system so it persists across Streamlit reruns."""
    from main import DrugRepurposingSystem
    system = DrugRepurposingSystem(
        enable_dgem=enable_dgem,
        enable_pathway=enable_pathway,
        enable_proximity=enable_proximity,
        enable_literature=enable_literature,
    )
    return system



# ─── Lottie loading overlay ───
LOTTIE_URL = "https://assets6.lottiefiles.com/private_files/lf30_ghyv7l.json"


def show_loading_overlay(
    disease_name: str,
    status_text: str = "Synthesizing repurposing candidates",
):
    """Inject a fullscreen Lottie overlay into the parent Streamlit page."""
    safe_name = disease_name.replace("&", "&amp;").replace("<", "&lt;")
    safe_status = (
        status_text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    lottie_src = LOTTIE_URL

    # The iframe's only job is to run JS that creates an overlay in the
    # parent Streamlit document.  When Streamlit clears the placeholder
    # the iframe is removed, and a MutationObserver auto-removes the overlay.
    injection_html = f"""
    <script>
    (function() {{
      var doc = window.parent.document;

      // Prevent duplicates
      if (doc.getElementById('drug-loading-overlay')) return;

      // ── Inject lottie-player script into parent if not present ──
      if (!doc.querySelector('script[data-lottie-loader]')) {{
        var s = doc.createElement('script');
        s.src = 'https://unpkg.com/@lottiefiles/lottie-player@2.0.8/dist/lottie-player.js';
        s.setAttribute('data-lottie-loader', '1');
        doc.head.appendChild(s);
      }}

      // ── Detect dark theme ──
      var html = doc.documentElement;
      var isDark = html.getAttribute('data-theme') === 'dark'
        || html.classList.contains('dark')
        || doc.querySelector('[data-testid="stAppViewContainer"]')
              ?.getAttribute('data-theme') === 'dark'
        || window.matchMedia('(prefers-color-scheme:dark)').matches;

      // ── Create overlay ──
      var ov = doc.createElement('div');
      ov.id = 'drug-loading-overlay';

      var bg   = isDark ? 'rgba(14,17,23,0.88)' : 'rgba(255,255,255,0.84)';
      var txt   = isDark ? '#e0e0e0' : '#333';

      ov.innerHTML = `
        <style>
          #drug-loading-overlay {{
            position: fixed; inset: 0; z-index: 999999;
            display: flex; flex-direction: column;
            align-items: center; justify-content: center; gap: 1.2rem;
            background: ${{bg}};
            backdrop-filter: blur(18px) saturate(1.4);
            -webkit-backdrop-filter: blur(18px) saturate(1.4);
            animation: ov-fade-in 0.35s ease-out;
            font-family: 'Source Sans Pro','Segoe UI',Roboto,sans-serif;
            color: ${{txt}};
          }}
          @keyframes ov-fade-in {{
            from {{ opacity: 0; }} to {{ opacity: 1; }}
          }}

          #drug-loading-overlay lottie-player {{
            width:  min(380px, 60vw);
            height: min(380px, 60vw);
            filter: drop-shadow(0 4px 32px rgba(99,102,241,0.2));
          }}

          .dl-disease-tag {{
            display: inline-block;
            padding: 0.5rem 1.6rem;
            border-radius: 999px;
            font-size: 1.15rem; font-weight: 700;
            letter-spacing: 0.03em;
            background: linear-gradient(135deg, #6366f1, #a855f7);
            color: #fff;
            box-shadow: 0 3px 20px rgba(99,102,241,0.3);
          }}

          .dl-status {{
            font-size: 1.2rem; font-weight: 600;
            opacity: 0.78;
          }}
          .dl-status::after {{
            content: '';
            animation: dl-dots 1.5s steps(1) infinite;
          }}
          @keyframes dl-dots {{
            0%,20%  {{ content: ''; }}
            40%     {{ content: '.'; }}
            60%     {{ content: '..'; }}
            80%,100%{{ content: '...'; }}
          }}

          .dl-prog-track {{
            width: min(340px, 55vw); height: 5px;
            border-radius: 3px;
            background: ${{isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.06)'}};
            overflow: hidden;
          }}
          .dl-prog-fill {{
            height: 100%; border-radius: 3px;
            background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
            animation: dl-sweep 2.8s ease-in-out infinite alternate;
          }}
          @keyframes dl-sweep {{
            0%   {{ width: 5%; }}
            100% {{ width: 95%; }}
          }}
        </style>

        <lottie-player
          src="{lottie_src}"
          background="transparent"
          speed="1" autoplay loop>
        </lottie-player>
        <div class="dl-disease-tag">{safe_name}</div>
        <div class="dl-status">{safe_status}</div>
        <div class="dl-prog-track"><div class="dl-prog-fill"></div></div>
      `;

      doc.body.appendChild(ov);

      // ── Auto-remove overlay when Streamlit clears the placeholder ──
      // Watch for this iframe being removed from the DOM
      var me = window.frameElement;
      if (me) {{
        var observer = new MutationObserver(function(mutations) {{
          if (!doc.contains(me)) {{
            var el = doc.getElementById('drug-loading-overlay');
            if (el) el.remove();
            observer.disconnect();
          }}
        }});
        observer.observe(doc.body, {{ childList: true, subtree: true }});
      }}
    }})();
    </script>
    """
    components.html(injection_html, height=0, scrolling=False)


def hide_loading_overlay():
    """Remove the loading overlay from the parent Streamlit page."""
    components.html("""
    <script>
    (function() {
      var ov = window.parent.document.getElementById('drug-loading-overlay');
      if (ov) ov.remove();
    })();
    </script>
    """, height=0, scrolling=False)


st.title("🔬 AI Drug Repurposing System")
st.caption("DGEM + Pathway + Proximity + Literature")

# ─── Sidebar ───
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of drug candidates", 1, 20, 10)
    explain_top = st.slider(
        "DGEM drugs to explain in detail", 0, 20, 10,
        help="Biological explanations use the top DGEM-ranked drugs only.",
    )

    st.divider()
    st.header("Scoring Modules")
    enable_dgem = st.checkbox("DGEM (Gene Expression)", value=True)
    enable_pathway = st.checkbox("Pathway Enrichment", value=True)
    enable_proximity = st.checkbox("Network Proximity", value=True)
    enable_literature = st.checkbox("Literature Mining", value=True)

    st.divider()
    _busy = st.session_state.get("_init_busy", False)
    _init_clicked = False
    _ic1, _ic2 = st.columns([1, 0.2], gap="small")
    with _ic1:
        if _busy:
            st.button("Initialize System", disabled=True, key="init_btn_busy")
        else:
            _init_clicked = st.button("Initialize System", key="init_btn")
    with _ic2:
        if _busy:
            st.markdown(
                "<div style='display:flex;align-items:center;height:38px'>"
                "<span style='font-size:1.35rem;line-height:1;"
                "animation:initpulse 1s ease-in-out infinite'>⏳</span></div>"
                "<style>@keyframes initpulse{"
                "0%,100%{opacity:1}50%{opacity:0.25}}</style>",
                unsafe_allow_html=True,
            )

    if not _busy and _init_clicked:
        st.session_state["_init_busy"] = True
        st.session_state["_init_kwargs"] = {
            "enable_dgem": enable_dgem,
            "enable_pathway": enable_pathway,
            "enable_proximity": enable_proximity,
            "enable_literature": enable_literature,
        }
        st.rerun()

    if _busy:
        _kw = st.session_state.pop("_init_kwargs", None)
        try:
            if _kw:
                st.session_state.system = load_system(
                    _kw["enable_dgem"],
                    _kw["enable_pathway"],
                    _kw["enable_proximity"],
                    _kw["enable_literature"],
                )
                st.session_state["_init_message"] = ("success", "System ready!")
        except Exception as _e:
            st.session_state["_init_message"] = ("error", f"Initialization failed: {_e}")
        finally:
            st.session_state["_init_busy"] = False
        st.rerun()

    # Show deferred init message after rerun
    _msg = st.session_state.pop("_init_message", None)
    if _msg:
        if _msg[0] == "success":
            st.success(_msg[1])
        else:
            st.error(_msg[1])

# ─── Load disease names for autocomplete ───
@st.cache_data(show_spinner=False)
def load_disease_names(data_folder="./data"):
    """Load all disease names from node.csv for autocomplete."""
    nodes_file = os.path.join(data_folder, "node.csv")
    if not os.path.exists(nodes_file):
        return []
    # Read only the two needed columns — much faster for large files
    df = pd.read_csv(
        nodes_file, sep='\t', on_bad_lines='skip',
        usecols=lambda c: c.strip('"') in ("node_type", "node_name"),
        dtype=str,
    )
    df.columns = [c.strip('"') for c in df.columns]
    df["node_name"] = df["node_name"].str.strip('"')
    df["node_type"] = df["node_type"].str.strip('"')
    diseases = df[df["node_type"] == "disease"]["node_name"].dropna().tolist()
    return sorted(set(diseases))


# ─── Two-pass disease list load: render page first, populate on rerun ───
# Pass 1: session_state has no list yet → show empty selectbox, load, rerun
# Pass 2: list is ready → show full selectbox (instant, from cache)
if "disease_list" not in st.session_state:
    st.session_state["disease_list"] = []
    st.session_state["_loading_diseases"] = True

if st.session_state.get("_loading_diseases"):
    st.session_state["disease_list"] = load_disease_names()
    st.session_state["_loading_diseases"] = False
    st.rerun()

disease_list = st.session_state["disease_list"]


# ─── Shared helper: render scoring module tabs ───
def _render_scoring_tabs(results):
    """Render Pathway / Proximity prediction tabs."""
    other_modules = [
        ("Pathway scorer", "pathway_predictions",
         "Pathway enrichment — top list scaled to 40-74% "
         "(score_raw in JSON)"),
        ("Shortest path - Network proximity", "proximity_predictions",
         "PPI network distance — top list scaled to 40-80% "
         "(score_raw in JSON)"),
    ]

    other_tab_names = []
    other_tab_data = []
    for name, key, desc in other_modules:
        preds = results.get(key, [])
        if preds:
            other_tab_names.append(name)
            other_tab_data.append((preds, desc, key))

    if not other_tab_names:
        return

    st.divider()
    st.subheader("Other Scoring Modules")
    tabs = st.tabs(other_tab_names)
    for tab, (preds, desc, key) in zip(tabs, other_tab_data):
        with tab:
            st.caption(desc)

            for i, pred in enumerate(preds):
                score = pred["score"]
                pct = score * 100
                color = ("green" if score > 0.7
                         else "orange" if score > 0.4
                         else "red")

                detail = ""

                if key == "pathway_predictions" and pred.get("pathways"):
                    pw_names = pred["pathways"]
                    detail = "  \n" + "  \n".join(
                        f"  *{pw}*" for pw in pw_names
                    )

                if key == "proximity_predictions":
                    parts = []
                    if pred.get("targets"):
                        targets_str = ", ".join(pred["targets"][:5])
                        parts.append(f"Targets: {targets_str}")
                    if "shortest_path" in pred:
                        parts.append(
                            f"d = {pred['shortest_path']}"
                        )
                    if parts:
                        detail = " | ".join(parts)
                        detail = f"  \n  *{detail}*"

                if key == "literature_predictions":
                    extra_parts = []
                    if pred.get("pubmed_count"):
                        extra_parts.append(
                            f"{pred['pubmed_count']} papers"
                        )
                    citations = pred.get("citations", [])
                    if citations:
                        cite_str = "; ".join(citations[:3])
                        extra_parts.append(cite_str)
                    if extra_parts:
                        detail = f"  \n  *{' | '.join(extra_parts)}*"

                st.markdown(
                    f"**{i+1}. {pred['drug']}** — "
                    f":{color}[{pct:.0f}%]{detail}"
                )


# ─── Main Area: Top-level tabs ───
tab_repurpose, tab_research = st.tabs(
    ["Drug Repurposing", "Research PDF Analysis"]
)

# ════════════════════════════════════════════════════════════
# Tab 1: Drug Repurposing (existing functionality)
# ════════════════════════════════════════════════════════════
with tab_repurpose:
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
            # ── Fullscreen Lottie loading overlay ──
            loading_placeholder = st.empty()
            with loading_placeholder:
                show_loading_overlay(disease_input)

            try:
                results = st.session_state.system.repurpose(
                    disease_input,
                    top_k=top_k,
                    explain_top=explain_top
                )
            except Exception as e:
                loading_placeholder.empty()
                hide_loading_overlay()
                st.error(f"Error: {e}")
                st.stop()

            loading_placeholder.empty()
            hide_loading_overlay()

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

            # Literature & DGEM side by side
            lit_preds = results.get("literature_predictions", [])
            dgem_preds = results.get("dgem_predictions", [])

            if lit_preds or dgem_preds:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Literature mining (40% for the number + 60% for the quality)")
                    st.caption("PubMed co-occurrence + LLM relevance")
                    if lit_preds:
                        for i, pred in enumerate(lit_preds):
                            score = pred["score"]
                            pct = score * 100
                            color = ("green" if score > 0.7
                                     else "orange" if score > 0.4
                                     else "red")
                            detail = ""
                            extra_parts = []
                            if pred.get("pubmed_count"):
                                extra_parts.append(
                                    f"{pred['pubmed_count']} papers"
                                )
                            citations = pred.get("citations", [])
                            if citations:
                                cite_str = "; ".join(citations[:3])
                                extra_parts.append(cite_str)
                            if extra_parts:
                                detail = (f"  \n  *"
                                          f"{' | '.join(extra_parts)}*")
                            st.markdown(
                                f"**{i+1}. {pred['drug']}** — "
                                f":{color}[{pct:.0f}%]{detail}"
                            )
                    else:
                        st.info("No Literature predictions.")

                with col2:
                    st.subheader("DGEM Predictions")
                    st.caption(
                        "Gene expression reversal — top list scaled to 80-84% "
                        "(see score_raw in JSON for raw 0-1 reversal)"
                    )
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

            # Other modules in tabs
            _render_scoring_tabs(results)

            # Explanations (DGEM top 5 only, in DGEM rank order)
            expl_map = results.get("explanations") or {}
            if expl_map and dgem_preds:
                st.divider()
                st.subheader("Biological Explanations (DGEM Top Candidates)")
                for p in dgem_preds[:5]:
                    drug = p["drug"]
                    key = next(
                        (k for k in expl_map if k.lower() == drug.lower()),
                        None,
                    )
                    if key is None:
                        continue
                    expl = expl_map[key]
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
                rs = (results.get("metadata") or {}).get("report_source", "")
                if rs == "dgem":
                    st.subheader("Full Report (DGEM)")
                    st.caption(
                        "Narrative is grounded in the gene expression reversal "
                        "ranking, not TxGNN graph scores."
                    )
                else:
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

# ════════════════════════════════════════════════════════════
# Tab 2: Research PDF Analysis
# ════════════════════════════════════════════════════════════
with tab_research:
    st.header("Research PDF Analysis")
    st.caption(
        "Upload a research paper PDF to extract disease information, "
        "gene expressions, and pathways, then search for drug candidates."
    )

    uploaded_pdf = st.file_uploader(
        "Upload a research PDF",
        type=["pdf"],
        key="research_pdf",
    )

    # Overwrite control for existing diseases
    overwrite_existing = st.checkbox(
        "Overwrite existing disease signature if it already exists",
        value=False,
        key="overwrite_sig",
    )

    if st.button("Analyze PDF", type="primary", key="analyze_pdf_btn"):
        if "system" not in st.session_state:
            st.error("Initialize the system first (sidebar button).")
        elif uploaded_pdf is None:
            st.warning("Please upload a PDF file first.")
        else:
            system = st.session_state.system

            # Check if DGEM is available for signature saving
            dgem_available = (system.dgem is not None
                              and system.dgem._initialized)

            loading_pdf = st.empty()
            with loading_pdf:
                show_loading_overlay(
                    uploaded_pdf.name or "Research PDF",
                    status_text="Extracting information from PDF...",
                )
            try:
                from research_module import ResearchPDFAnalyzer
                analyzer = ResearchPDFAnalyzer(system.nlp, system.dgem)
                analysis = analyzer.analyze(uploaded_pdf)
            except Exception as e:
                loading_pdf.empty()
                hide_loading_overlay()
                st.error(f"PDF extraction failed: {e}")
                st.stop()

            loading_pdf.empty()
            hide_loading_overlay()

            extracted = analysis["extracted_info"]
            disease_name = extracted.get("disease_name", "")
            gene_expressions = extracted.get("gene_expressions", [])
            pathways_found = extracted.get("pathways", [])
            summary = extracted.get("summary", "")
            alignment_stats = analysis["alignment_stats"]

            if not disease_name:
                st.error("Could not extract a disease name from the PDF.")
                st.stop()

            # ── Display extracted info ──
            st.header(f"Extracted: {disease_name}")

            with st.expander("Extracted Information", expanded=True):
                st.write(f"**Disease:** {disease_name}")
                st.write(f"**Summary:** {summary}")
                if pathways_found:
                    st.write(f"**Pathways:** {', '.join(pathways_found)}")
                st.write(f"**Genes found:** {len(gene_expressions)}")

                if gene_expressions:
                    import pandas as pd  # noqa: F811
                    gene_df = pd.DataFrame(gene_expressions)
                    st.dataframe(gene_df, use_container_width=True)

            # ── Gene alignment stats ──
            if gene_expressions:
                with st.expander("Gene Expression Alignment"):
                    matched = alignment_stats.get("matched_l1000", 0)
                    unmatched = alignment_stats.get("unmatched", 0)
                    total = alignment_stats.get("total_extracted", 0)

                    st.write(f"**Total genes extracted:** {total}")
                    st.write(f"**Matched to L1000 panel:** {matched}")
                    st.write(f"**Not in L1000 panel:** {unmatched}")

                    if matched == 0:
                        st.warning(
                            "No extracted genes matched the L1000 gene panel. "
                            "DGEM scoring will not be informative."
                        )

                    unmatched_genes = alignment_stats.get("unmatched_genes", [])
                    if unmatched_genes:
                        st.caption(
                            f"Unmatched genes: {', '.join(unmatched_genes)}"
                        )

            # ── Save signature and run scoring ──
            # Check for existing disease
            skip_save = False
            if (dgem_available and disease_name
                    and analysis["signature_vector"] is not None):
                disease_lower = disease_name.lower().strip()
                existing = (system.dgem.disease_signatures
                            and disease_lower in system.dgem.disease_signatures)
                if existing and not overwrite_existing:
                    st.warning(
                        f"Disease '{disease_name}' already has a signature "
                        "in the database. Check 'Overwrite existing' above "
                        "to replace it."
                    )
                    skip_save = True

            loading_score = st.empty()
            with loading_score:
                show_loading_overlay(
                    disease_name,
                    status_text="Running drug scoring modules...",
                )
            try:
                results = system.analyze_research_pdf(
                    top_k=top_k,
                    save_signature=not skip_save,
                    pre_extracted=analysis,
                )
            except Exception as e:
                loading_score.empty()
                hide_loading_overlay()
                st.error(f"Analysis failed: {e}")
                st.stop()

            loading_score.empty()
            hide_loading_overlay()

            # ── Signature save status ──
            if results.get("signature_saved"):
                st.success(
                    f"Gene expression signature saved for '{disease_name}' "
                    f"({alignment_stats.get('matched_l1000', 0)} L1000 genes)."
                )

            # ── DGEM predictions ──
            dgem_preds = results.get("dgem_predictions", [])
            if dgem_preds:
                st.divider()
                st.subheader("DGEM Predictions")
                st.caption(
                    "Gene expression reversal scoring based on the "
                    "extracted signature"
                )
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

            # ── Other scoring modules ──
            _render_scoring_tabs(results)

            # ── Report ──
            if results.get("report"):
                st.divider()
                st.subheader("Full Report")
                st.markdown(results["report"])

            # ── Metadata ──
            if results.get("metadata"):
                with st.expander("Metadata"):
                    st.json(results["metadata"])

            # ── Download ──
            st.download_button(
                "Download Full Results (JSON)",
                json.dumps(results, indent=2, default=str),
                file_name=f"research_{disease_name.replace(' ', '_')}.json",
                mime="application/json",
                key="download_research",
            )
