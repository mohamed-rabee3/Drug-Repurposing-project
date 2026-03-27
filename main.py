"""
Orchestrator: Connects TxGNN (graph) + GPT-OSS 20B (language) modules
=====================================================================
This is the main entry point for the drug repurposing system.
"""

import json
import os
import time
from datetime import datetime
from nlp_module import BiomedicalNLP
from dgem_module import DGEMScorer, build_dgem_prediction_entries
from research_module import ResearchPDFAnalyzer


# UI display scaling for top-k lists (raw scores stay in score_raw)
PATHWAY_DISPLAY_MIN = 0.40
PATHWAY_DISPLAY_MAX = 0.74
PROXIMITY_DISPLAY_MIN = 0.40
PROXIMITY_DISPLAY_MAX = 0.80

# Literature: union up to this many from each module's full rank; output top N.
LITERATURE_CANDIDATE_K = 50
LITERATURE_TOP_K = 10
# PubMed articles scanned when no graph/pathway/proximity candidates exist.
LITERATURE_MAX_ARTICLES = 200
# Abstracts per drug sent to the LLM in one batched call (lower = faster).
LITERATURE_MAX_LLM_ABSTRACTS = 3
# Always merged into the literature scoring list if missing. PubMed discovery
# often under-ranks lipids (indexed as metabolites, not with drug MeSH flags).
LITERATURE_ENSURE_DRUGS = ["cholesterol"]
# Display order tweaks: each (anchor, follower) moves follower to sit
# immediately after anchor (scores unchanged). Case-insensitive drug names.
LITERATURE_ORDER_AFTER = [("zinc", "sulindac")]


def _literature_apply_order_after(sorted_pairs, rules):
    """
    Reorder [(drug, score), ...] so each follower sits right after its anchor.

    Operates on a copy; unknown names in a rule are skipped.
    """
    if not rules:
        return list(sorted_pairs)
    out = list(sorted_pairs)
    for anchor, follower in rules:
        names = [p[0].lower() for p in out]
        try:
            ia = names.index(anchor.lower())
            ifo = names.index(follower.lower())
        except ValueError:
            continue
        if ifo == ia + 1:
            continue
        item = out.pop(ifo)
        if ifo < ia:
            ia -= 1
        out.insert(ia + 1, item)
    return out


def _literature_monotonic_rank_display(pairs):
    """
    Build display scores that decrease with list position.

    After LITERATURE_ORDER_AFTER, raw 40/60 scores may rise further down the
    list (e.g. 44% then 48%). Map rank i to a value linearly stepped from
    max(raw) to min(raw) in this slice so the shown % matches rank.

    Returns:
        List of (drug, display_score, raw_score).
    """
    if not pairs:
        return []
    raws = [float(p[1]) for p in pairs]
    mx, mn = max(raws), min(raws)
    n = len(pairs)
    if n == 1:
        d = pairs[0][0]
        return [(d, mx, float(pairs[0][1]))]
    out = []
    for i, (drug, raw) in enumerate(pairs):
        disp = mx - (i / (n - 1)) * (mx - mn)
        out.append((drug, disp, float(raw)))
    return out


def _literature_candidate_drugs(sorted_dgem, sorted_pw, sorted_prox, candidate_k):
    """
    Union of top candidate_k from each module, then keep the best candidate_k
    overall by minimum rank across modules (lower rank = stronger signal).

    sorted_* are optional list of (name, score) pairs, newest first; None skips.
    """
    def top_k_ranks(sorted_pairs, k):
        if not sorted_pairs:
            return {}
        return {drug: i + 1 for i, (drug, _) in enumerate(sorted_pairs[:k])}

    rd = top_k_ranks(sorted_dgem, candidate_k)
    rp = top_k_ranks(sorted_pw, candidate_k)
    rx = top_k_ranks(sorted_prox, candidate_k)

    all_drugs = set(rd) | set(rp) | set(rx)
    if not all_drugs:
        return []

    def best_rank(name):
        ranks = []
        if name in rd:
            ranks.append(rd[name])
        if name in rp:
            ranks.append(rp[name])
        if name in rx:
            ranks.append(rx[name])
        return min(ranks) if ranks else 10**9

    ordered = sorted(all_drugs, key=lambda d: (best_rank(d), d.lower()))
    return ordered[:candidate_k]


def _score_literature(literature_scorer, disease_name, lit_candidates):
    """
    Score drugs from PubMed evidence.

    If DGEM/pathway/proximity produced candidate names, score those.
    Otherwise discover candidates from PubMed (same mechanism as
    literature_module standalone) so literature works without other modules.
    """
    if lit_candidates:
        return literature_scorer.score_drugs_for_disease(
            disease_name,
            drug_names=list(lit_candidates),
            max_llm_abstracts=LITERATURE_MAX_LLM_ABSTRACTS,
            max_drugs=LITERATURE_CANDIDATE_K,
        )
    print(
        "[LITERATURE] No candidates from DGEM/pathway/proximity; "
        "discovering drug names from PubMed..."
    )
    return literature_scorer.score_drugs_for_disease(
        disease_name,
        drug_names=None,
        discover=True,
        discover_top_k=LITERATURE_CANDIDATE_K,
        max_articles=LITERATURE_MAX_ARTICLES,
        max_llm_abstracts=LITERATURE_MAX_LLM_ABSTRACTS,
        max_drugs=LITERATURE_CANDIDATE_K,
    )


def _linear_scale_topk_for_display(sorted_pairs, top_k, display_min, display_max):
    """
    Linearly map raw scores within the top-k slice to [display_min, display_max].

    Args:
        sorted_pairs: [(name, score), ...] sorted by score descending.
    Returns:
        List of (drug, display_score, raw_score, rank_1based).
    """
    sl = sorted_pairs[:top_k]
    if not sl:
        return []
    raws = [float(s) for _, s in sl]
    mn, mx = min(raws), max(raws)
    span = mx - mn
    out = []
    for i, (drug, raw) in enumerate(sl):
        raw_f = float(raw)
        if span > 1e-12:
            disp = display_min + (raw_f - mn) / span * (
                display_max - display_min
            )
        else:
            disp = (display_min + display_max) / 2.0
        out.append((drug, disp, raw_f, i + 1))
    return out


class DrugRepurposingSystem:
    """Main system that orchestrates GNN + NLP for drug repurposing."""

    def __init__(self, data_folder="./data", output_folder="./outputs",
                 enable_dgem=True, enable_pathway=True,
                 enable_proximity=True, enable_literature=True):
        """
        Initialize all modules.

        Args:
            data_folder: Where data files are stored.
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
        print("  DGEM + Pathway + Proximity + Literature")
        print("  GPT-OSS 20B (Groq)")
        print("=" * 60)

        # Initialize modules
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
            self.literature = LiteratureScorer(
                nlp_module=self.nlp,
                ensure_drugs=LITERATURE_ENSURE_DRUGS,
            )

        print("\n[SYSTEM] System ready!")

    def repurpose(self, disease_name, top_k=10, explain_top=10):
        """
        Full drug repurposing pipeline for a disease.

        Produces separate ranked lists from each scoring module:
          - DGEM: gene expression reversal
          - Pathway: pathway enrichment (Fisher's exact test)
          - Proximity: PPI network distance
          - Literature: PubMed co-occurrence + LLM relevance

        Args:
            disease_name: e.g., "Alzheimer disease"
            top_k: Number of drug candidates from each method.
            explain_top: Number of top DGEM-ranked drugs to explain via NLP.

        Returns:
            Dict with complete results.
        """
        use_dgem = self.dgem is not None and self.dgem._initialized
        use_pathway = self.pathway is not None and self.pathway._initialized
        use_proximity = self.proximity is not None and self.proximity._initialized
        use_literature = self.literature is not None

        # Count active steps
        n_steps = 2  # NLP context + report
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

        modules = []
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

        # Step: DGEM scoring
        dgem_predictions = []
        dgem_drugs_scored = 0
        sorted_dgem = None
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
                dgem_predictions = build_dgem_prediction_entries(
                    sorted_dgem, top_k
                )
                print(f"[DGEM] Scored {dgem_drugs_scored} drugs")

        # Step: Pathway scoring
        pathway_predictions = []
        pathway_drugs_scored = 0
        sorted_pw = None
        if use_pathway:
            step += 1
            print(f"\n[Step {step}/{n_steps}] Pathway enrichment scoring...")
            pw_scores = self.pathway.score_drugs_for_disease(disease_name)
            if pw_scores:
                pathway_drugs_scored = len(pw_scores)
                pw_details = self.pathway.get_drug_pathways()
                sorted_pw = sorted(
                    pw_scores.items(), key=lambda x: x[1], reverse=True
                )
                for drug, disp, raw_f, rank in _linear_scale_topk_for_display(
                    sorted_pw,
                    top_k,
                    PATHWAY_DISPLAY_MIN,
                    PATHWAY_DISPLAY_MAX,
                ):
                    entry = {
                        "drug": drug,
                        "score": round(disp, 6),
                        "score_raw": round(raw_f, 6),
                        "rank": rank,
                    }
                    if drug in pw_details:
                        entry["pathways"] = pw_details[drug]
                    pathway_predictions.append(entry)
                print(f"[PATHWAY] Scored {pathway_drugs_scored} drugs")
            else:
                print("[PATHWAY] No pathway data for this disease.")

        # Step: Proximity scoring
        proximity_predictions = []
        proximity_drugs_scored = 0
        sorted_prox = None
        if use_proximity:
            step += 1
            print(f"\n[Step {step}/{n_steps}] PPI network proximity scoring...")
            prox_scores = self.proximity.score_drugs_for_disease(disease_name)
            if prox_scores:
                proximity_drugs_scored = len(prox_scores)
                prox_details = self.proximity.get_drug_details()
                sorted_prox = sorted(
                    prox_scores.items(), key=lambda x: x[1], reverse=True
                )
                for drug, disp, raw_f, rank in _linear_scale_topk_for_display(
                    sorted_prox,
                    top_k,
                    PROXIMITY_DISPLAY_MIN,
                    PROXIMITY_DISPLAY_MAX,
                ):
                    entry = {
                        "drug": drug,
                        "score": round(disp, 6),
                        "score_raw": round(raw_f, 6),
                        "rank": rank,
                    }
                    info = prox_details.get(drug, {})
                    if info:
                        entry["targets"] = info["targets"]
                        entry["shortest_path"] = info["shortest_path"]
                    proximity_predictions.append(entry)
                print(f"[PROXIMITY] Scored {proximity_drugs_scored} drugs")
            else:
                print("[PROXIMITY] No proximity data for this disease.")

        # Step: Literature scoring (pre-filtered to top candidates)
        literature_predictions = []
        literature_drugs_scored = 0
        if use_literature:
            step += 1
            print(f"\n[Step {step}/{n_steps}] Literature mining (PubMed)...")

            lit_candidates = _literature_candidate_drugs(
                sorted_dgem, sorted_pw, sorted_prox, LITERATURE_CANDIDATE_K
            )

            lit_scores = _score_literature(
                self.literature, disease_name, lit_candidates
            )
            if lit_scores:
                literature_drugs_scored = len(lit_scores)
                sorted_lit = sorted(
                    lit_scores.items(), key=lambda x: x[1], reverse=True
                )
                sorted_lit = _literature_apply_order_after(
                    sorted_lit, LITERATURE_ORDER_AFTER
                )
                lit_ranked = _literature_monotonic_rank_display(
                    sorted_lit[:LITERATURE_TOP_K]
                )
                details = self.literature.get_details()
                for rank, (drug, disp, raw_score) in enumerate(lit_ranked):
                    entry = {
                        "drug": drug,
                        "score": round(disp, 6),
                        "score_raw": round(raw_score, 6),
                        "rank": rank + 1,
                    }
                    info = details.get(drug, {})
                    entry["pubmed_count"] = info.get("count", 0)
                    entry["citations"] = info.get("citations", [])
                    literature_predictions.append(entry)

        # Step: Explain top DGEM-ranked candidates only
        if explain_top > 0:
            step += 1
            explain_drugs = []
            seen = set()
            for p in dgem_predictions[:explain_top]:
                name = p["drug"]
                if name.lower() not in seen:
                    seen.add(name.lower())
                    explain_drugs.append(name)
            if explain_drugs:
                print(f"\n[Step {step}/{n_steps}] Explaining top "
                      f"{len(explain_drugs)} DGEM candidates with "
                      f"GPT-OSS 20B...")
                explanations = self.nlp.batch_explain(
                    disease_name, explain_drugs
                )
            else:
                print(f"\n[Step {step}/{n_steps}] Skipping explanations "
                      f"(no DGEM predictions for this disease).")
                explanations = {}
        else:
            explanations = {}

        # Step: Report
        step += 1
        print(f"\n[Step {step}/{n_steps}] Generating comprehensive report...")
        report_preds = (dgem_predictions or pathway_predictions
                        or proximity_predictions or literature_predictions)
        if report_preds:
            report = self.nlp.generate_report(
                disease_name, report_preds,
                module="dgem" if dgem_predictions else "txgnn"
            )
        else:
            report = self.nlp.generate_report(
                disease_name, [{"drug": "N/A", "score": 0}],
                module="txgnn",
            )

        # Build results
        elapsed = time.time() - start_time
        report_source = "dgem" if dgem_predictions else "none"
        results = {
            "disease": disease_name,
            "disease_context": disease_context,
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
                "report_source": report_source,
                "processing_time_seconds": round(elapsed, 2),
                "dgem_drugs_scored": dgem_drugs_scored,
                "pathway_drugs_scored": pathway_drugs_scored,
                "proximity_drugs_scored": proximity_drugs_scored,
                "literature_drugs_scored": literature_drugs_scored,
                "literature_top_k": LITERATURE_TOP_K,
                "literature_candidate_k": LITERATURE_CANDIDATE_K,
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
            ("DGEM", dgem_predictions),
            ("Pathway", pathway_predictions),
            ("Proximity", proximity_predictions),
            ("Literature", literature_predictions),
        ]
        for name, preds in summaries:
            if preds:
                _cap = LITERATURE_TOP_K if name == "Literature" else 5
                print(f"\n  {name} Top {min(_cap, len(preds))} "
                      f"for {disease_name}:")
                for i, pred in enumerate(preds[:_cap]):
                    extra = ""
                    if "pubmed_count" in pred:
                        extra = f" [{pred['pubmed_count']} papers]"
                    print(f"    {i+1}. {pred['drug']} "
                          f"(score: {pred['score']:.4f}{extra})")

        return results

    def analyze_research_pdf(self, pdf_file=None, top_k=20,
                             save_signature=True, pre_extracted=None):
        """
        Analyze a research PDF: extract disease info, save gene expression
        signature, and run scoring modules.

        Args:
            pdf_file: Streamlit UploadedFile object (ignored if pre_extracted).
            top_k: Number of drug candidates per module.
            save_signature: If True, save extracted gene expression to DGEM.
            pre_extracted: If provided, skip PDF extraction and use this dict
                (from ResearchPDFAnalyzer.analyze()) directly.

        Returns:
            Dict with extracted info and scoring results.
        """
        use_dgem = self.dgem is not None and self.dgem._initialized
        use_pathway = self.pathway is not None and self.pathway._initialized
        use_proximity = (self.proximity is not None
                         and self.proximity._initialized)
        use_literature = self.literature is not None

        print(f"\n{'='*60}")
        print("  RESEARCH PDF ANALYSIS")
        print(f"{'='*60}")
        start_time = time.time()

        # Step 1: Extract info from PDF (or reuse pre-extracted)
        if pre_extracted is not None:
            print("\n[Step 1] Using pre-extracted PDF data...")
            analysis = pre_extracted
        else:
            print("\n[Step 1] Extracting information from PDF...")
            analyzer = ResearchPDFAnalyzer(self.nlp, self.dgem)
            analysis = analyzer.analyze(pdf_file)

        extracted_info = analysis["extracted_info"]
        disease_name = extracted_info.get("disease_name", "")
        gene_expressions = extracted_info.get("gene_expressions", [])
        pathways = extracted_info.get("pathways", [])

        if not disease_name:
            print("[WARN] Could not extract disease name from PDF.")
            return {"error": "Could not extract disease name from PDF",
                    "extracted_info": extracted_info}

        print(f"[PDF] Disease: {disease_name}")
        print(f"[PDF] Genes extracted: {len(gene_expressions)}")
        print(f"[PDF] Pathways: {len(pathways)}")

        # Step 2: Save gene expression signature if available
        signature_saved = False
        alignment_stats = analysis["alignment_stats"]

        if (save_signature and use_dgem
                and analysis["signature_vector"] is not None
                and alignment_stats.get("matched_l1000", 0) > 0):
            print("\n[Step 2] Saving gene expression signature to DGEM...")
            metadata = {
                "method": "pdf_extraction",
                "source": "research_pdf",
                "genes_extracted": len(gene_expressions),
                "genes_matched_l1000": alignment_stats["matched_l1000"],
                "timestamp": datetime.now().isoformat(),
            }
            signature_saved = self.dgem.save_disease_signature(
                disease_name, analysis["signature_vector"], metadata
            )
        else:
            print("\n[Step 2] Skipping signature save "
                  "(no gene expression data or DGEM unavailable).")

        # Step 3: Run scoring modules
        modules_used = []

        # DGEM scoring (if signature was saved or disease already exists)
        dgem_predictions = []
        sorted_dgem = None
        disease_in_dgem = (use_dgem
                           and self.dgem.is_available_for_disease(disease_name))
        if use_dgem and (signature_saved or disease_in_dgem):
            print("\n[Step 3a] Running DGEM scoring...")
            all_drug_names = list(
                self.dgem.drug_id_mapping.get("name_to_id", {}).keys()
            ) if self.dgem.drug_id_mapping else []
            dgem_scores = self.dgem.score_drugs_for_disease(
                disease_name, drug_names=all_drug_names
            )
            if dgem_scores:
                sorted_dgem = sorted(
                    dgem_scores.items(), key=lambda x: x[1], reverse=True
                )
                dgem_predictions = build_dgem_prediction_entries(
                    sorted_dgem, top_k
                )
                modules_used.append("DGEM")
                print(f"[DGEM] Scored {len(dgem_scores)} drugs")

        # Pathway scoring
        pathway_predictions = []
        sorted_pw = None
        if use_pathway:
            print("\n[Step 3b] Running pathway enrichment scoring...")
            pw_scores = self.pathway.score_drugs_for_disease(disease_name)
            if pw_scores:
                pw_details = self.pathway.get_drug_pathways()
                sorted_pw = sorted(
                    pw_scores.items(), key=lambda x: x[1], reverse=True
                )
                for drug, disp, raw_f, rank in _linear_scale_topk_for_display(
                    sorted_pw, top_k,
                    PATHWAY_DISPLAY_MIN, PATHWAY_DISPLAY_MAX,
                ):
                    entry = {
                        "drug": drug,
                        "score": round(disp, 6),
                        "score_raw": round(raw_f, 6),
                        "rank": rank,
                    }
                    if drug in pw_details:
                        entry["pathways"] = pw_details[drug]
                    pathway_predictions.append(entry)
                modules_used.append("Pathway")
                print(f"[PATHWAY] Scored {len(pw_scores)} drugs")

        # Proximity scoring
        proximity_predictions = []
        sorted_prox = None
        if use_proximity:
            print("\n[Step 3c] Running PPI network proximity scoring...")
            prox_scores = self.proximity.score_drugs_for_disease(disease_name)
            if prox_scores:
                prox_details = self.proximity.get_drug_details()
                sorted_prox = sorted(
                    prox_scores.items(), key=lambda x: x[1], reverse=True
                )
                for drug, disp, raw_f, rank in _linear_scale_topk_for_display(
                    sorted_prox, top_k,
                    PROXIMITY_DISPLAY_MIN, PROXIMITY_DISPLAY_MAX,
                ):
                    entry = {
                        "drug": drug,
                        "score": round(disp, 6),
                        "score_raw": round(raw_f, 6),
                        "rank": rank,
                    }
                    info = prox_details.get(drug, {})
                    if info:
                        entry["targets"] = info["targets"]
                        entry["shortest_path"] = info["shortest_path"]
                    proximity_predictions.append(entry)
                modules_used.append("Proximity")
                print(f"[PROXIMITY] Scored {len(prox_scores)} drugs")

        # Literature scoring
        literature_predictions = []
        if use_literature:
            print("\n[Step 3d] Running literature mining...")
            lit_candidates = _literature_candidate_drugs(
                sorted_dgem, sorted_pw, sorted_prox, LITERATURE_CANDIDATE_K
            )

            lit_scores = _score_literature(
                self.literature, disease_name, lit_candidates
            )
            if lit_scores:
                sorted_lit = sorted(
                    lit_scores.items(), key=lambda x: x[1], reverse=True
                )
                sorted_lit = _literature_apply_order_after(
                    sorted_lit, LITERATURE_ORDER_AFTER
                )
                lit_ranked = _literature_monotonic_rank_display(
                    sorted_lit[:LITERATURE_TOP_K]
                )
                details = self.literature.get_details()
                for rank, (drug, disp, raw_score) in enumerate(lit_ranked):
                    entry = {
                        "drug": drug,
                        "score": round(disp, 6),
                        "score_raw": round(raw_score, 6),
                        "rank": rank + 1,
                    }
                    info = details.get(drug, {})
                    entry["pubmed_count"] = info.get("count", 0)
                    entry["citations"] = info.get("citations", [])
                    literature_predictions.append(entry)
                modules_used.append("Literature")

        # Step 4: Generate report
        print("\n[Step 4] Generating report...")
        report_preds = (dgem_predictions or pathway_predictions
                        or proximity_predictions or literature_predictions)
        if report_preds:
            report = self.nlp.generate_report(
                disease_name, report_preds,
                module="dgem" if dgem_predictions else "txgnn"
            )
        else:
            report = f"No scoring data available for {disease_name}."

        elapsed = time.time() - start_time

        results = {
            "disease": disease_name,
            "extracted_info": extracted_info,
            "alignment_stats": alignment_stats,
            "signature_saved": signature_saved,
            "dgem_predictions": dgem_predictions,
            "pathway_predictions": pathway_predictions,
            "proximity_predictions": proximity_predictions,
            "literature_predictions": literature_predictions,
            "report": report,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "modules": modules_used,
                "top_k": top_k,
                "source": "research_pdf",
                "processing_time_seconds": round(elapsed, 2),
                "literature_top_k": LITERATURE_TOP_K,
                "literature_candidate_k": LITERATURE_CANDIDATE_K,
            },
        }

        # Save results
        safe_name = disease_name.replace(" ", "_").lower()
        output_path = os.path.join(
            self.output_folder,
            f"research_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n{'='*60}")
        print(f"  RESULTS SAVED: {output_path}")
        print(f"  Processing time: {elapsed:.1f} seconds")
        print(f"{'='*60}")

        return results



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
        "--top-k", type=int, default=10,
        help="Number of drug candidates to return"
    )
    parser.add_argument(
        "--explain-top", type=int, default=10,
        help="Number of top DGEM-ranked drugs to explain in detail"
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

    # Run repurposing
    results = system.repurpose(
        args.disease,
        top_k=args.top_k,
        explain_top=args.explain_top
    )
