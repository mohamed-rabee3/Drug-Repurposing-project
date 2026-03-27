"""
Module 2: NLP Layer using GPT-OSS 20B via Groq API
====================================================
This module handles:
  - Biomedical literature understanding
  - Generating explanations for drug-disease predictions
  - Extracting structured knowledge from text
  - Enriching GNN predictions with biological context
"""

import os
import json
import re
import time
import hashlib
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class BiomedicalNLP:
    """NLP module for biomedical text understanding via Groq API."""

    def __init__(self, cache_folder="./cache"):
        """
        Initialize the NLP module.

        Args:
            cache_folder: Where to cache API responses to avoid repeated calls.
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_actual_api_key_here":
            raise ValueError(
                "GROQ_API_KEY not found or not configured! "
                "Create a .env file with: GROQ_API_KEY=your_key_here\n"
                "Get your key at: https://console.groq.com"
            )

        self.client = Groq(api_key=api_key)
        self.model = "openai/gpt-oss-20b"
        self.cache_folder = cache_folder
        try:
            os.makedirs(cache_folder, exist_ok=True)
        except (FileExistsError, OSError):
            pass

        print(f"[NLP] Initialized with model: {self.model}")
        print(f"[NLP] Cache folder: {cache_folder}")

    def _call_groq(self, system_prompt, user_prompt, temperature=0.3,
                    max_tokens=2000, use_cache=True, max_retries=3):
        """
        Make a call to the Groq API with caching and retry logic.

        Args:
            system_prompt: Instructions for the model's behavior.
            user_prompt: The actual question/task.
            temperature: 0.0 = deterministic, 1.0 = creative.
            max_tokens: Maximum response length.
            use_cache: If True, check cache before making API call.
            max_retries: Number of retries on rate limit errors.

        Returns:
            String response from the model.
        """
        # Check cache first
        raw = f"{system_prompt}{user_prompt}{temperature}".encode('utf-8')
        cache_key = hashlib.md5(raw).hexdigest()
        cache_file = os.path.join(self.cache_folder, f"{cache_key}.json")

        if use_cache and os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            return cached["response"]

        # Make API call with retry
        for attempt in range(max_retries):
            try:
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=temperature,
                        max_completion_tokens=max_tokens,
                    )
                except TypeError:
                    # Older groq SDK versions use max_tokens
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                response = completion.choices[0].message.content

                # Save to cache
                if use_cache:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump({
                            "system_prompt": system_prompt[:200],
                            "user_prompt": user_prompt[:200],
                            "response": response,
                            "timestamp": time.time()
                        }, f, indent=2)

                return response

            except Exception as e:
                err_str = str(e).lower()
                if ("rate_limit" in err_str or "429" in err_str) and attempt < max_retries - 1:
                    wait = 2 ** attempt * 5  # 5s, 10s, 20s
                    print(f"[NLP] Rate limited. Waiting {wait}s "
                          f"(attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait)
                else:
                    print(f"[NLP] API Error: {e}")
                    return f"Error: {str(e)}"

    def _parse_json_response(self, response):
        """
        Robustly extract JSON from LLM response text.

        Handles markdown fences, trailing commas, and extra text.
        """
        text = response.strip()
        # Remove markdown code fences
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?\s*```$', '', text)
        text = text.strip()
        # Remove trailing commas before } or ]
        text = re.sub(r',\s*([}\]])', r'\1', text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Last resort: find the first { ... } block
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    cleaned = re.sub(r',\s*([}\]])', r'\1', match.group())
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    pass
            return {
                "summary": response,
                "mechanism_of_action": "See summary",
                "evidence_strength": "unknown",
                "parse_error": True
            }

    def explain_drug_disease_relationship(self, drug_name, disease_name):
        """
        Generate a biological explanation for why a drug might treat a disease.

        Args:
            drug_name: e.g., "Metformin"
            disease_name: e.g., "Alzheimer disease"

        Returns:
            Dict with structured explanation.
        """
        system_prompt = (
            "You are a biomedical research assistant specializing "
            "in drug repurposing. You explain drug-disease relationships using "
            "established biological knowledge. Always be accurate and cite "
            "known mechanisms.\n\n"
            "Respond ONLY in valid JSON format with these fields:\n"
            "{\n"
            '    "mechanism_of_action": "How the drug works at the molecular level",\n'
            '    "biological_pathway": "Which biological pathway connects the drug to the disease",\n'
            '    "target_proteins": ["List", "of", "key", "protein", "targets"],\n'
            '    "evidence_strength": "strong / moderate / weak / theoretical",\n'
            '    "known_studies": "Brief mention of any known research supporting this connection",\n'
            '    "safety_considerations": "Key safety notes for this repurposing",\n'
            '    "summary": "2-3 sentence plain-English summary a doctor could understand"\n'
            "}"
        )

        user_prompt = (
            f'Explain why the drug "{drug_name}" could potentially '
            f'be repurposed to treat "{disease_name}".\n\n'
            "Focus on:\n"
            "1. The drug's known mechanism of action\n"
            "2. The biological pathways involved in the disease\n"
            "3. How the drug's mechanism could affect those pathways\n"
            "4. Any known evidence from literature or clinical trials\n\n"
            "If there is no known connection, say so honestly and explain "
            "what the theoretical basis might be."
        )

        response = self._call_groq(system_prompt, user_prompt)
        return self._parse_json_response(response)

    def enrich_disease_context(self, disease_name):
        """
        Get rich biomedical context about a disease for better predictions.

        Args:
            disease_name: e.g., "Parkinson disease"

        Returns:
            Dict with disease context information.
        """
        system_prompt = (
            "You are a biomedical knowledge expert. "
            "Provide structured information about diseases for drug repurposing research.\n\n"
            "Respond ONLY in valid JSON format:\n"
            "{\n"
            '    "disease_name": "Official name",\n'
            '    "disease_category": "e.g., neurodegenerative, autoimmune, metabolic",\n'
            '    "key_pathways": ["pathway1", "pathway2"],\n'
            '    "key_targets": ["protein1", "gene1"],\n'
            '    "current_treatments": ["drug1", "drug2"],\n'
            '    "unmet_needs": "What current treatments fail to address",\n'
            '    "similar_diseases": ["disease1", "disease2"],\n'
            '    "pathophysiology_summary": "Brief description of disease mechanism"\n'
            "}"
        )

        user_prompt = (
            f'Provide detailed biomedical context for the disease: "{disease_name}"\n\n'
            "Focus on information useful for drug repurposing:\n"
            "- Key molecular targets\n"
            "- Biological pathways involved\n"
            "- Current treatment gaps\n"
            "- Diseases with similar mechanisms"
        )

        response = self._call_groq(system_prompt, user_prompt)
        result = self._parse_json_response(response)
        if "disease_name" not in result and not result.get("parse_error"):
            result["disease_name"] = disease_name
        return result

    def generate_report(self, disease_name, drug_predictions,
                        module="txgnn"):
        """
        Generate a comprehensive drug repurposing report.

        Args:
            disease_name: The target disease.
            drug_predictions: List of dicts with at least 'drug' and 'score'.
                For DGEM entries, 'score_raw' (0-1 reversal) is preferred
                in the prompt when present.
            module: 'dgem' (gene expression reversal) or 'txgnn' (GNN).

        Returns:
            String — a formatted markdown report.
        """
        lines = []
        for i, d in enumerate(drug_predictions[:10]):
            drug = d.get("drug", "Unknown")
            if module == "dgem" and "score_raw" in d:
                s = float(d["score_raw"])
                lines.append(
                    f"  {i+1}. {drug} (DGEM raw reversal score 0-1: {s:.3f})"
                )
            elif "score_raw" in d:
                s = float(d["score_raw"])
                lines.append(
                    f"  {i+1}. {drug} (raw score 0-1: {s:.3f})"
                )
            else:
                s = float(d.get("score", 0))
                label = (
                    "GNN confidence (0-1)"
                    if module == "txgnn"
                    else "score (0-1)"
                )
                lines.append(f"  {i+1}. {drug} ({label}: {s:.3f})")
        drug_list = "\n".join(lines)

        system_prompt = (
            "You are a clinical research consultant writing a "
            "drug repurposing report. Write clearly for both scientists and "
            "clinicians. Use markdown formatting. Be factual and balanced — "
            "note limitations."
        )

        if module == "dgem":
            user_prompt = (
                f"Write a drug repurposing report for:\n\n"
                f"Disease: {disease_name}\n\n"
                "Methodology to describe in the report: candidates were ranked "
                "by Drug–Gene Expression Matching (DGEM): each drug's "
                "expression perturbation profile (L1000-style) is compared to "
                "the disease signature; higher scores indicate stronger "
                "predicted reversal of disease-associated expression (Connectivity "
                "Map hypothesis). This is orthogonal to target-based GNN scores — "
                "do NOT refer to Graph Neural Networks or GNN confidence for "
                "these candidates.\n\n"
                f"DGEM-ranked drug candidates:\n{drug_list}\n\n"
                "In the report title and tables, use terminology such as "
                "'DGEM reversal score' or 'gene expression match' — never "
                "'GNN' or 'graph neural network' for this ranking.\n\n"
                "For each of the top 5 drugs, include:\n"
                "1. Known mechanism of action\n"
                "2. Why it might work for this disease\n"
                "3. Existing evidence (if any)\n"
                "4. Safety considerations\n"
                "5. Recommended next steps\n\n"
                "End with an overall summary and limitations section. Clearly "
                "state that DGEM scores reflect computational expression "
                "concordance, not clinical efficacy or safety."
            )
        else:
            user_prompt = (
                f"Write a drug repurposing report for:\n\n"
                f"Disease: {disease_name}\n\n"
                f"AI-predicted drug candidates (ranked by GNN confidence score):\n"
                f"{drug_list}\n\n"
                "For each of the top 5 drugs, include:\n"
                "1. Known mechanism of action\n"
                "2. Why it might work for this disease\n"
                "3. Existing evidence (if any)\n"
                "4. Safety considerations\n"
                "5. Recommended next steps\n\n"
                "End with an overall summary and limitations section."
            )

        return self._call_groq(
            system_prompt, user_prompt,
            temperature=0.4,
            max_tokens=4000
        )

    def batch_explain(self, disease_name, drug_list, max_drugs=5):
        """
        Explain multiple drug-disease relationships efficiently.

        Args:
            disease_name: Target disease.
            drug_list: List of drug names to explain.
            max_drugs: Maximum number to explain (to control API costs).

        Returns:
            Dict mapping drug names to their explanations.
        """
        results = {}
        for drug in drug_list[:max_drugs]:
            print(f"[NLP] Explaining: {drug} -> {disease_name}...")
            results[drug] = self.explain_drug_disease_relationship(
                drug, disease_name
            )
            time.sleep(0.5)  # Rate limiting
        return results


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing NLP Module...")
    nlp = BiomedicalNLP()

    # Test 1: Disease context
    print("\n--- Test 1: Disease Context ---")
    context = nlp.enrich_disease_context("Alzheimer disease")
    print(json.dumps(context, indent=2))

    # Test 2: Drug-disease explanation
    print("\n--- Test 2: Drug-Disease Explanation ---")
    explanation = nlp.explain_drug_disease_relationship(
        "Metformin", "Alzheimer disease"
    )
    print(json.dumps(explanation, indent=2))

    print("\nNLP Module test complete!")
