"""
Module 8: Literature Mining
============================
Scores drug-disease pairs by PubMed co-occurrence and LLM-based
relevance scoring of abstracts.

Data source: PubMed E-utilities API (free, no auth required)
LLM: Groq GPT-OSS 20B (via existing nlp_module)
"""

import os
import json
import hashlib
import time
import requests
import xml.etree.ElementTree as ET
import numpy as np


class LiteratureScorer:
    """Score drugs by literature evidence from PubMed."""

    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def __init__(self, nlp_module=None, cache_folder="./cache/literature",
                 api_key=None):
        """
        Args:
            nlp_module: BiomedicalNLP instance for LLM relevance scoring.
            cache_folder: Where to cache PubMed results.
            api_key: NCBI API key (optional, increases rate to 10 req/sec).
        """
        self.nlp = nlp_module
        self.cache_folder = cache_folder
        os.makedirs(cache_folder, exist_ok=True)

        self.api_key = api_key or os.environ.get("NCBI_API_KEY")
        self._rate_delay = 0.11 if self.api_key else 0.34  # 10/sec or 3/sec
        self._initialized = True

        rate = "10/sec" if self.api_key else "3/sec"
        llm = "yes" if self.nlp else "count-only"
        print(f"[LITERATURE] Ready. PubMed rate: {rate}, LLM scoring: {llm}")

    def _cache_path(self, drug_name, disease_name, suffix):
        key = f"{drug_name.lower()}|{disease_name.lower()}|{suffix}"
        h = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_folder, f"{h}.json")

    def _search_pubmed(self, drug_name, disease_name):
        """Search PubMed for drug-disease co-occurrence. Returns (count, pmids)."""
        cache_path = self._cache_path(drug_name, disease_name, "search")
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                cached = json.load(f)
            return cached["count"], cached["pmids"]

        params = {
            "db": "pubmed",
            "term": f'"{drug_name}"[TIAB] AND "{disease_name}"[TIAB]',
            "retmax": 20,
            "retmode": "json",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            time.sleep(self._rate_delay)
            resp = requests.get(self.ESEARCH_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            result = data.get("esearchresult", {})
            count = int(result.get("count", 0))
            pmids = result.get("idlist", [])
        except Exception:
            count = 0
            pmids = []

        # Cache
        with open(cache_path, "w") as f:
            json.dump({"count": count, "pmids": pmids}, f)

        return count, pmids

    def _fetch_abstracts(self, pmids, drug_name, disease_name):
        """Fetch abstracts from PubMed for given PMIDs."""
        if not pmids:
            return []

        cache_path = self._cache_path(drug_name, disease_name, "abstracts")
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return json.load(f)

        # Fetch up to 10 abstracts
        fetch_pmids = pmids[:10]
        params = {
            "db": "pubmed",
            "id": ",".join(fetch_pmids),
            "rettype": "abstract",
            "retmode": "xml",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        abstracts = []
        try:
            time.sleep(self._rate_delay)
            resp = requests.get(self.EFETCH_URL, params=params, timeout=30)
            resp.raise_for_status()

            root = ET.fromstring(resp.content)
            for article in root.findall(".//PubmedArticle"):
                title_el = article.find(".//ArticleTitle")
                title = title_el.text if title_el is not None and title_el.text else ""

                abstract_parts = []
                for text_el in article.findall(".//AbstractText"):
                    if text_el.text:
                        abstract_parts.append(text_el.text)
                abstract_text = " ".join(abstract_parts)

                pmid_el = article.find(".//PMID")
                pmid = pmid_el.text if pmid_el is not None else ""

                if abstract_text:
                    abstracts.append({
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract_text[:2000],
                    })
        except Exception:
            pass

        with open(cache_path, "w") as f:
            json.dump(abstracts, f)

        return abstracts

    def _score_abstract_relevance(self, drug_name, disease_name, abstract):
        """Use LLM to score abstract relevance for drug repurposing (0-10)."""
        if not self.nlp:
            return 5.0  # Neutral if no LLM

        prompt = (
            f"Rate how strongly this abstract supports using {drug_name} "
            f"to treat {disease_name}. Score 0-10 where:\n"
            f"  0 = completely irrelevant\n"
            f"  5 = mentions both but no therapeutic link\n"
            f"  10 = strong evidence of therapeutic benefit\n\n"
            f"Title: {abstract['title']}\n"
            f"Abstract: {abstract['abstract'][:1500]}\n\n"
            f"Reply with ONLY a JSON object: {{\"score\": <number>, "
            f"\"reason\": \"<one sentence>\"}}"
        )

        try:
            response = self.nlp._call_groq(
                system_prompt="You are a biomedical literature reviewer.",
                user_prompt=prompt,
                temperature=0.1,
                max_tokens=200,
                use_cache=True
            )
            parsed = self.nlp._parse_json_response(response)
            if isinstance(parsed, dict) and "score" in parsed:
                return float(parsed["score"])
        except Exception:
            pass

        return 5.0

    def score_drugs_for_disease(self, disease_name, drug_names=None, top_k=20):
        """
        Score drugs by literature evidence.

        Args:
            disease_name: Disease name.
            drug_names: List of drug names to score.
            top_k: Number of top results to return.

        Returns:
            Dict mapping drug_name -> literature_score (0-1).
        """
        if not drug_names:
            return {}

        print(f"[LITERATURE] Searching PubMed for {len(drug_names)} drugs "
              f"x '{disease_name}'...")

        raw_scores = {}
        details = {}

        for i, drug_name in enumerate(drug_names):
            if (i + 1) % 50 == 0:
                print(f"  ...searched {i+1}/{len(drug_names)}")

            count, pmids = self._search_pubmed(drug_name, disease_name)

            if count == 0:
                continue

            # Co-occurrence score (log-scaled)
            cooccurrence = np.log10(count + 1)

            # LLM relevance scoring for drugs with papers
            llm_score = 0.0
            abstracts = self._fetch_abstracts(pmids, drug_name, disease_name)
            if abstracts and self.nlp:
                scores = []
                for abstract in abstracts[:5]:
                    s = self._score_abstract_relevance(
                        drug_name, disease_name, abstract
                    )
                    scores.append(s)
                llm_score = np.mean(scores) / 10.0  # Normalize to [0, 1]

            raw_scores[drug_name] = {
                "cooccurrence": cooccurrence,
                "llm_relevance": llm_score,
                "count": count,
                "abstracts_analyzed": len(abstracts),
            }

        if not raw_scores:
            return {}

        # Normalize co-occurrence across all scored drugs
        cooc_values = [v["cooccurrence"] for v in raw_scores.values()]
        max_cooc = max(cooc_values)
        min_cooc = min(cooc_values)
        cooc_range = max_cooc - min_cooc

        results = {}
        for drug, info in raw_scores.items():
            if cooc_range > 0:
                norm_cooc = (info["cooccurrence"] - min_cooc) / cooc_range
            else:
                norm_cooc = 1.0

            # Combine: 40% co-occurrence + 60% LLM relevance
            if self.nlp and info["llm_relevance"] > 0:
                score = 0.4 * norm_cooc + 0.6 * info["llm_relevance"]
            else:
                score = norm_cooc

            results[drug] = score
            details[drug] = info

        self._last_details = details

        hits = sum(1 for v in raw_scores.values() if v["count"] > 0)
        print(f"[LITERATURE] Found publications for {hits}/{len(drug_names)} drugs")

        return results

    def get_details(self):
        """Get detailed scoring info from last run."""
        return getattr(self, "_last_details", {})

    def get_coverage_stats(self):
        return {
            "initialized": self._initialized,
            "has_llm": self.nlp is not None,
            "has_api_key": self.api_key is not None,
        }


if __name__ == "__main__":
    scorer = LiteratureScorer()
    disease = "fragile x syndrome"
    drugs = ["Sulindac", "Metformin", "Propranolol", "Aripiprazole",
             "Donepezil", "Lithium", "Minocycline"]

    print(f"\nSearching PubMed for {len(drugs)} drugs x '{disease}'...")
    scores = scorer.score_drugs_for_disease(disease, drug_names=drugs)

    if scores:
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        print(f"\nResults:")
        details = scorer.get_details()
        for drug, score in ranked:
            info = details.get(drug, {})
            print(f"  {drug}: score={score:.4f}, "
                  f"papers={info.get('count', 0)}")
    else:
        print("No results found.")
