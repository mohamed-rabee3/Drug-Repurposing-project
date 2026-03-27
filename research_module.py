"""
Module: Research PDF Analysis
==============================
Extracts disease information (name, gene expressions, pathways) from uploaded
research PDFs using the Groq LLM, builds L1000-aligned gene expression
signatures, and prepares data for scoring by existing modules.
"""

import numpy as np
import pdfplumber


class ResearchPDFAnalyzer:
    """Extracts disease info from research PDFs using LLM."""

    def __init__(self, nlp_module, dgem_module=None):
        """
        Args:
            nlp_module: BiomedicalNLP instance (for LLM calls).
            dgem_module: DGEMScorer instance (for L1000 gene list and saving).
        """
        self.nlp = nlp_module
        self.dgem = dgem_module

    def extract_text_from_pdf(self, uploaded_file):
        """
        Extract text from a PDF file.

        Args:
            uploaded_file: Streamlit UploadedFile or file path string.

        Returns:
            str: Extracted text (truncated to ~15,000 chars).

        Raises:
            ValueError: If no text could be extracted.
        """
        pages_text = []
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)

        if not pages_text:
            raise ValueError(
                "Could not extract text from PDF. "
                "The file may be a scanned image or empty."
            )

        full_text = "\n\n".join(pages_text)
        # Truncate for LLM context window
        max_chars = 15000
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars]

        return full_text

    def extract_disease_info(self, pdf_text):
        """
        Use LLM to extract structured disease information from PDF text.

        Args:
            pdf_text: Raw text extracted from the research PDF.

        Returns:
            Dict with disease_name, gene_expressions, pathways, summary.
        """
        system_prompt = (
            "You are a biomedical research expert. Extract structured "
            "information from research paper text.\n\n"
            "Respond ONLY in valid JSON format with these fields:\n"
            "{\n"
            '    "disease_name": "the canonical disease name studied in the paper",\n'
            '    "gene_expressions": [\n'
            '        {"gene": "HGNC_symbol", "direction": "up" or "down", '
            '"fold_change": number or null}\n'
            "    ],\n"
            '    "pathways": ["pathway names mentioned or implicated"],\n'
            '    "summary": "brief 2-3 sentence summary of the paper findings"\n'
            "}\n\n"
            "Rules:\n"
            "- Use official HGNC gene symbols (e.g., TP53, BRCA1, EGFR)\n"
            "- For gene_expressions, include ALL genes mentioned as "
            "differentially expressed, upregulated, or downregulated\n"
            "- direction must be exactly 'up' or 'down'\n"
            "- fold_change should be a number if reported, null if not\n"
            "- Include pathways even if only mentioned in discussion\n"
            "- disease_name should be the primary disease, not comorbidities"
        )

        user_prompt = (
            "Extract the disease name, differentially expressed genes "
            "(with direction and fold change if available), biological "
            "pathways, and a brief summary from this research paper:\n\n"
            f"{pdf_text}"
        )

        response = self.nlp._call_groq(
            system_prompt, user_prompt,
            temperature=0.1,
            max_tokens=3000,
            use_cache=False,
        )
        return self.nlp._parse_json_response(response)

    def build_signature_vector(self, extracted_genes):
        """
        Build an L1000-aligned gene expression signature vector.

        Args:
            extracted_genes: List of dicts with gene, direction, fold_change.

        Returns:
            Tuple of (np.ndarray of shape (n_l1000_genes,), stats dict).
        """
        if not self.dgem or not self.dgem.l1000_genes:
            return None, {"error": "DGEM module or L1000 gene list not available"}

        l1000_genes = self.dgem.l1000_genes
        n_genes = len(l1000_genes)

        # Build case-insensitive lookup
        gene_to_idx = {g.upper(): i for i, g in enumerate(l1000_genes)}

        vector = np.zeros(n_genes, dtype=np.float32)
        matched = []
        unmatched = []

        for entry in extracted_genes:
            gene = entry.get("gene", "").strip()
            if not gene:
                continue

            direction = entry.get("direction", "up").lower()
            fold_change = entry.get("fold_change")

            # Default magnitude
            magnitude = float(fold_change) if fold_change is not None else 1.0
            magnitude = abs(magnitude)

            # Apply direction
            value = magnitude if direction == "up" else -magnitude

            idx = gene_to_idx.get(gene.upper())
            if idx is not None:
                vector[idx] = value
                matched.append(gene)
            else:
                unmatched.append(gene)

        stats = {
            "total_extracted": len(extracted_genes),
            "matched_l1000": len(matched),
            "unmatched": len(unmatched),
            "matched_genes": matched,
            "unmatched_genes": unmatched,
            "nonzero_in_vector": int(np.count_nonzero(vector)),
        }

        return vector, stats

    def analyze(self, uploaded_file):
        """
        Full analysis pipeline: extract text -> extract info -> build vector.

        Args:
            uploaded_file: Streamlit UploadedFile object.

        Returns:
            Dict with extracted_info, signature_vector, alignment_stats.
        """
        # Step 1: Extract text
        pdf_text = self.extract_text_from_pdf(uploaded_file)

        # Step 2: Extract disease info via LLM
        extracted_info = self.extract_disease_info(pdf_text)

        # Step 3: Build signature vector if genes were found
        gene_expressions = extracted_info.get("gene_expressions", [])
        signature_vector = None
        alignment_stats = {"total_extracted": 0, "matched_l1000": 0}

        if gene_expressions and self.dgem:
            signature_vector, alignment_stats = self.build_signature_vector(
                gene_expressions
            )

        return {
            "extracted_info": extracted_info,
            "signature_vector": signature_vector,
            "alignment_stats": alignment_stats,
            "pdf_text_length": len(pdf_text),
        }
