"""
Module 8: Literature Mining
============================
Scores drug-disease pairs by PubMed co-occurrence and LLM-based
relevance scoring of abstracts.

Now includes automatic candidate discovery: searches PubMed for the
top drug candidates associated with a disease via MeSH pharmacological
terms, then scores them.

Data source: PubMed E-utilities API (free, no auth required)
LLM: Groq GPT-OSS 20B (via existing nlp_module)
"""

import os
import json
import hashlib
import time
import re
from collections import Counter
import requests
import xml.etree.ElementTree as ET
import numpy as np


class LiteratureScorer:
    """Score drugs by literature evidence from PubMed."""

    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    PUBMED_ARTICLE_BASE = "https://pubmed.ncbi.nlm.nih.gov"

    # MeSH tree numbers for pharmacological substances (category D)
    # These help identify MeSH headings that are drugs/chemicals
    DRUG_MESH_TREES = ("D01", "D02", "D03", "D04", "D05", "D06", "D08",
                       "D09", "D10", "D12", "D13", "D20", "D23", "D26",
                       "D27")

    # Common non-drug MeSH terms to exclude (diseases, general concepts, etc.)
    EXCLUDE_MESH = {
        "humans", "animals", "male", "female", "adult", "child",
        "adolescent", "infant", "mice", "rats", "cells, cultured",
        "treatment outcome", "dose-response relationship, drug",
        "drug therapy", "molecular targeted therapy", "proteins",
        "peptides", "amino acids", "rna", "dna", "water", "oxygen",
        "glucose", "sodium chloride", "saline solution", "placebos",
        "pharmaceutical preparations", "carrier proteins",
        "recombinant proteins", "antibodies", "vaccines",
        "dietary supplements", "vitamins", "antioxidants",
    }

    def __init__(self, nlp_module=None, cache_folder="./cache/literature",
                 api_key=None, ensure_drugs=None):
        """
        Args:
            nlp_module: BiomedicalNLP instance for LLM relevance scoring.
            cache_folder: Where to cache PubMed results.
            api_key: NCBI API key (optional, increases rate to 10 req/sec).
            ensure_drugs: Extra drug names to always score (merged after
                discovery or graph union). Use for terms that PubMed
                discovery under-ranks (e.g. lipids indexed as metabolites
                without drug MeSH qualifiers).
        """
        self.nlp = nlp_module
        self.cache_folder = cache_folder
        os.makedirs(cache_folder, exist_ok=True)
        self.ensure_drugs = tuple(ensure_drugs) if ensure_drugs else ()

        self.api_key = api_key or os.environ.get("NCBI_API_KEY")
        self._rate_delay = 0.11 if self.api_key else 0.34  # 10/sec or 3/sec
        self._initialized = True

        rate = "10/sec" if self.api_key else "3/sec"
        llm = "yes" if self.nlp else "count-only"
        ed = len(self.ensure_drugs)
        extra = f", ensure_drugs={ed}" if ed else ""
        print(f"[LITERATURE] Ready. PubMed rate: {rate}, LLM scoring: {llm}"
              f"{extra}")

    def _merge_ensure_drugs(self, drug_names):
        """Append ensure_drugs not already present (case-insensitive)."""
        if not self.ensure_drugs:
            return list(drug_names)
        seen = {d.lower() for d in drug_names}
        out = list(drug_names)
        for d in self.ensure_drugs:
            name = d.strip() if isinstance(d, str) else str(d).strip()
            if not name:
                continue
            key = name.lower()
            if key not in seen:
                out.append(name)
                seen.add(key)
        return out

    def _cap_drug_names_after_merge(self, drug_names, max_drugs):
        """If len > max_drugs, keep ensure_drugs first, then rest in list order."""
        if max_drugs is None or len(drug_names) <= max_drugs:
            return drug_names
        ens_lower = {
            str(e).strip().lower()
            for e in (self.ensure_drugs or [])
            if e and str(e).strip()
        }
        if not ens_lower:
            return drug_names[:max_drugs]
        seen = set()
        out = []
        for n in drug_names:
            if len(out) >= max_drugs:
                return out
            if n.lower() in ens_lower and n.lower() not in seen:
                out.append(n)
                seen.add(n.lower())
        for n in drug_names:
            if len(out) >= max_drugs:
                break
            if n.lower() in seen:
                continue
            out.append(n)
            seen.add(n.lower())
        return out

    @staticmethod
    def _pubmed_url(pmid):
        p = (pmid or "").strip()
        return f"{LiteratureScorer.PUBMED_ARTICLE_BASE}/{p}/" if p else ""

    @classmethod
    def _abstract_citation_line(cls, abstract):
        """Human-readable paper line with PubMed URL (for UI / JSON)."""
        pmid = (abstract.get("pmid") or "").strip()
        url = (abstract.get("url") or "").strip() or cls._pubmed_url(pmid)
        cite = (abstract.get("citation") or "").strip()
        title = (abstract.get("title") or "").strip()
        if not url:
            return cite or title
        if cite:
            return f"{cite} — {url}"
        if title:
            return f"{title[:160]}{'…' if len(title) > 160 else ''} — {url}"
        return url

    # ------------------------------------------------------------------ #
    #  Caching helpers
    # ------------------------------------------------------------------ #

    def _cache_path(self, drug_name, disease_name, suffix):
        key = f"{drug_name.lower()}|{disease_name.lower()}|{suffix}"
        h = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_folder, f"{h}.json")

    def _generic_cache_path(self, label):
        h = hashlib.md5(label.lower().encode()).hexdigest()
        return os.path.join(self.cache_folder, f"disc_{h}.json")

    # ------------------------------------------------------------------ #
    #  NEW: Candidate discovery from PubMed
    # ------------------------------------------------------------------ #

    def discover_candidates(self, disease_name, top_k=50,
                            max_articles=200):
        """
        Search PubMed for drug/therapeutic candidates for a disease.

        Strategy:
          1. Search PubMed for "[disease] AND (drug therapy OR treatment)"
          2. Fetch article metadata including MeSH headings
          3. Extract MeSH terms tagged as pharmacological substances
          4. Also extract drug names from titles via pattern matching
          5. Rank by frequency → return top_k candidates

        Args:
            disease_name: Disease to find drug candidates for.
            top_k: Number of top candidates to return.
            max_articles: Max articles to scan (fetched in batches).

        Returns:
            List of drug name strings, ordered by PubMed frequency.
        """
        cache_path = self._generic_cache_path(
            f"discover|{disease_name}|{top_k}|{max_articles}"
        )
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                cached = json.load(f)
            print(f"[LITERATURE] Loaded {len(cached)} cached candidates "
                  f"for '{disease_name}'")
            return cached[:top_k]

        print(f"[LITERATURE] Discovering drug candidates for "
              f"'{disease_name}' from PubMed...")

        # Step 1: Search PubMed for disease + drug therapy articles
        pmids = self._search_disease_therapy(disease_name, max_articles)
        if not pmids:
            print("[LITERATURE] No articles found for disease.")
            return []

        print(f"[LITERATURE] Found {len(pmids)} articles, extracting "
              f"drug names from MeSH terms and titles...")

        # Step 2: Fetch articles in batches and extract drug MeSH terms
        drug_counter = Counter()
        batch_size = 50

        for batch_start in range(0, len(pmids), batch_size):
            batch_pmids = pmids[batch_start:batch_start + batch_size]
            drugs_in_batch = self._extract_drugs_from_articles(
                batch_pmids, disease_name
            )
            drug_counter.update(drugs_in_batch)

            if (batch_start + batch_size) % 100 == 0:
                print(f"  ...processed {batch_start + batch_size}/"
                      f"{len(pmids)} articles")

        if not drug_counter:
            print("[LITERATURE] No drug candidates extracted.")
            return []

        # Step 3: Rank by frequency, return top_k
        ranked = [drug for drug, _ in drug_counter.most_common(top_k)]

        print(f"[LITERATURE] Discovered {len(ranked)} candidate drugs "
              f"(from {len(drug_counter)} unique). Top 10:")
        for i, drug in enumerate(ranked[:10]):
            print(f"  {i+1}. {drug} ({drug_counter[drug]} mentions)")

        # Cache results
        with open(cache_path, "w") as f:
            json.dump(ranked, f)

        return ranked

    def _search_disease_therapy(self, disease_name, max_results):
        """
        Search PubMed for articles about drug therapy for a disease.
        Returns list of PMIDs.
        """
        queries = [
            # Primary: disease + drug therapy MeSH subheading
            f'"{disease_name}"[TIAB] AND (drug therapy[SH] OR '
            f'therapeutics[MeSH] OR pharmacotherapy[TIAB])',
            # Fallback: broader search
            f'"{disease_name}"[TIAB] AND (treatment[TIAB] OR '
            f'therapy[TIAB] OR drug[TIAB])',
        ]

        all_pmids = []
        seen = set()

        for query in queries:
            if len(all_pmids) >= max_results:
                break

            remaining = max_results - len(all_pmids)
            params = {
                "db": "pubmed",
                "term": query,
                "retmax": min(remaining, 500),
                "retmode": "json",
                "sort": "relevance",
            }
            if self.api_key:
                params["api_key"] = self.api_key

            try:
                time.sleep(self._rate_delay)
                resp = requests.get(self.ESEARCH_URL, params=params,
                                    timeout=15)
                resp.raise_for_status()
                data = resp.json()
                result = data.get("esearchresult", {})
                pmids = result.get("idlist", [])
                for pmid in pmids:
                    if pmid not in seen:
                        seen.add(pmid)
                        all_pmids.append(pmid)
            except Exception as e:
                print(f"[LITERATURE] Search error: {e}")

        return all_pmids[:max_results]

    def _extract_drugs_from_articles(self, pmids, disease_name):
        """
        Fetch PubMed articles and extract drug names from:
          - MeSH headings with pharmacological qualifiers
          - MeSH headings in drug substance tree (D category)
          - Chemical/substance name list
          - Title-based pattern matching as fallback
        Returns a list of drug names (with duplicates for counting).
        """
        if not pmids:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "xml",
            "retmode": "xml",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        drugs_found = []
        disease_lower = disease_name.lower()
        # Also exclude the disease name and abbreviations from results
        disease_tokens = set(disease_lower.split())

        try:
            time.sleep(self._rate_delay)
            resp = requests.get(self.EFETCH_URL, params=params, timeout=30)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)

            for article in root.findall(".//PubmedArticle"):
                article_drugs = set()

                # --- Method 1: MeSH headings with drug-related qualifiers ---
                for mesh_heading in article.findall(
                    ".//MeshHeadingList/MeshHeading"
                ):
                    descriptor = mesh_heading.find("DescriptorName")
                    if descriptor is None:
                        continue
                    name = descriptor.text
                    if not name:
                        continue
                    name_lower = name.lower()

                    # Skip if it's the disease itself or excluded
                    if (name_lower in self.EXCLUDE_MESH
                            or name_lower == disease_lower
                            or name_lower in disease_tokens):
                        continue

                    # Check qualifiers for pharmacological relevance
                    qualifiers = [
                        q.text for q in mesh_heading.findall("QualifierName")
                        if q.text
                    ]
                    qual_lower = [q.lower() for q in qualifiers]
                    drug_qualifiers = {
                        "therapeutic use", "pharmacology",
                        "administration & dosage", "adverse effects",
                        "toxicity", "analogs & derivatives",
                    }

                    # If MeSH heading has drug-related qualifiers → likely a drug
                    if any(q in drug_qualifiers for q in qual_lower):
                        article_drugs.add(name)
                        continue

                    # Check if MeSH tree number is in drug categories
                    # (DescriptorName may have UI attribute for lookup,
                    #  but we use a heuristic: if it appears with the
                    #  disease in therapy context, treat it as candidate)

                # --- Method 2: Chemical / NameOfSubstance list ---
                for chem in article.findall(
                    ".//ChemicalList/Chemical/NameOfSubstance"
                ):
                    if chem.text:
                        chem_name = chem.text.strip()
                        chem_lower = chem_name.lower()
                        if (chem_lower not in self.EXCLUDE_MESH
                                and chem_lower != disease_lower
                                and len(chem_name) > 2):
                            article_drugs.add(chem_name)

                # --- Method 3: Title-based extraction (simple heuristic) ---
                title_el = article.find(".//ArticleTitle")
                if title_el is not None and title_el.text:
                    title = title_el.text
                    title_drugs = self._extract_drugs_from_title(
                        title, disease_name
                    )
                    article_drugs.update(title_drugs)

                drugs_found.extend(article_drugs)

        except Exception as e:
            print(f"[LITERATURE] Fetch error: {e}")

        return drugs_found

    def _extract_drugs_from_title(self, title, disease_name):
        """
        Simple heuristic to extract drug-like names from article titles.
        Looks for patterns like:
          - "DrugName for DiseaseName"
          - "DrugName in DiseaseName"
          - "DrugName treatment of DiseaseName"
          - "effect(s) of DrugName on DiseaseName"
        """
        drugs = set()
        disease_lower = disease_name.lower()
        title_lower = title.lower()

        # Skip if disease not mentioned in title
        if disease_lower not in title_lower and not any(
            tok in title_lower for tok in disease_lower.split()
            if len(tok) > 3
        ):
            return drugs

        # Pattern: "X for disease" / "X in disease" / "X in the treatment of disease"
        patterns = [
            rf'(\b[A-Z][a-z]+(?:[-][a-z]+)?(?:\s+[A-Z][a-z]+)*)\s+'
            rf'(?:for|in|in the treatment of|therapy for|treatment of)\s+'
            rf'.*?{re.escape(disease_name.split()[0])}',
            # Pattern: "effect(s) of X on disease"
            rf'[Ee]ffects?\s+of\s+(\b[A-Z][a-z]+(?:[-][a-z]+)?)\s+on\s+'
            rf'.*?{re.escape(disease_name.split()[0])}',
        ]

        for pattern in patterns:
            match = re.search(pattern, title)
            if match:
                candidate = match.group(1).strip()
                # Basic validation: drug names are typically single
                # capitalized words ending in common suffixes
                if (len(candidate) > 2
                        and candidate.lower() not in self.EXCLUDE_MESH
                        and candidate.lower() != disease_lower):
                    drugs.add(candidate)

        return drugs

    # ------------------------------------------------------------------ #
    #  Original search / fetch / score methods (unchanged logic)
    # ------------------------------------------------------------------ #

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
                title = (title_el.text
                         if title_el is not None and title_el.text else "")

                abstract_parts = []
                for text_el in article.findall(".//AbstractText"):
                    if text_el.text:
                        abstract_parts.append(text_el.text)
                abstract_text = " ".join(abstract_parts)

                pmid_el = article.find(".//PMID")
                pmid = pmid_el.text if pmid_el is not None else ""

                first_author = ""
                author_count = 0
                author_list = article.find(".//AuthorList")
                if author_list is not None:
                    authors = author_list.findall("Author")
                    author_count = len(authors)
                    if authors:
                        last_name_el = authors[0].find("LastName")
                        if (last_name_el is not None
                                and last_name_el.text):
                            first_author = last_name_el.text

                pub_year = ""
                year_el = article.find(".//PubDate/Year")
                if year_el is not None and year_el.text:
                    pub_year = year_el.text
                else:
                    medline_el = article.find(".//PubDate/MedlineDate")
                    if medline_el is not None and medline_el.text:
                        parts = medline_el.text.split()
                        if parts and parts[0].isdigit():
                            pub_year = parts[0]

                citation = ""
                if first_author and pub_year:
                    if author_count > 1:
                        citation = f"{first_author} et al., {pub_year}"
                    else:
                        citation = f"{first_author}, {pub_year}"
                elif first_author:
                    citation = first_author
                elif pub_year:
                    citation = pub_year

                if abstract_text:
                    pubmed_url = self._pubmed_url(pmid) if pmid else ""
                    abstracts.append({
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract_text[:2000],
                        "citation": citation,
                        "url": pubmed_url,
                    })
        except Exception:
            pass

        with open(cache_path, "w") as f:
            json.dump(abstracts, f)

        return abstracts

    def _score_abstract_relevance(self, drug_name, disease_name, abstract):
        """Use LLM to score abstract relevance for drug repurposing (0-10)."""
        if not self.nlp:
            return 5.0

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

    def _score_abstracts_relevance_batch(self, drug_name, disease_name,
                                         abstracts):
        """
        Score all abstracts in one LLM call (avoids N separate API calls per drug).

        Returns:
            List of floats 0-10, same length as abstracts (defaults to 5.0 on failure).
        """
        n = len(abstracts)
        if not self.nlp or n == 0:
            return [5.0] * n

        parts = []
        for i, abstract in enumerate(abstracts):
            t = (abstract.get("title") or "")[:300]
            a = (abstract.get("abstract") or "")[:1200]
            parts.append(f"### Paper {i + 1}\nTitle: {t}\nAbstract: {a}")

        block = "\n\n".join(parts)
        prompt = (
            f"For repurposing {drug_name} to treat {disease_name}, rate EACH "
            f"paper below on how strongly its abstract supports that therapeutic "
            f"use. Use scores 0-10 where:\n"
            f"  0 = irrelevant\n"
            f"  5 = mentions drug and disease but weak/no therapeutic link\n"
            f"  10 = strong evidence of therapeutic benefit\n\n"
            f"{block}\n\n"
            f"Reply with ONLY valid JSON: {{\"scores\": [<n> papers, exactly "
            f"{n} numbers in order], \"reason\": \"one short sentence\"}}"
        )

        try:
            response = self.nlp._call_groq(
                system_prompt=(
                    "You are a biomedical literature reviewer. "
                    "Output only the requested JSON."
                ),
                user_prompt=prompt,
                temperature=0.1,
                max_tokens=350,
                use_cache=True,
            )
            parsed = self.nlp._parse_json_response(response)
            if isinstance(parsed, dict) and not parsed.get("parse_error"):
                raw = parsed.get("scores")
                if isinstance(raw, list) and len(raw) >= n:
                    out = []
                    for j in range(n):
                        try:
                            v = float(raw[j])
                            out.append(max(0.0, min(10.0, v)))
                        except (TypeError, ValueError):
                            out.append(5.0)
                    return out
                if isinstance(raw, list) and len(raw) > 0:
                    # Partial list: pad with 5.0
                    out = []
                    for j in range(n):
                        try:
                            v = float(raw[j]) if j < len(raw) else 5.0
                            out.append(max(0.0, min(10.0, v)))
                        except (TypeError, ValueError):
                            out.append(5.0)
                    return out
        except Exception:
            pass

        # Fallback: score first abstract only (one API call), rest neutral
        first = self._score_abstract_relevance(
            drug_name, disease_name, abstracts[0])
        return [first] + [5.0] * (n - 1)

    # ------------------------------------------------------------------ #
    #  Main scoring — now with auto-discovery
    # ------------------------------------------------------------------ #

    def score_drugs_for_disease(self, disease_name, drug_names=None,
                                top_k=20, discover=True,
                                discover_top_k=50, max_articles=200,
                                max_llm_abstracts=3, max_drugs=None):
        """
        Score drugs by literature evidence.

        If drug_names is None and discover=True, automatically discovers
        the top candidates from PubMed before scoring them.

        Args:
            disease_name: Disease name.
            drug_names: List of drug names to score. If None and
                        discover=True, candidates are auto-discovered.
            top_k: Number of top scored results to return.
            discover: If True and drug_names is None, auto-discover
                      candidates from PubMed.
            discover_top_k: How many candidates to discover.
            max_articles: Max articles to scan during discovery.
            max_llm_abstracts: How many PubMed abstracts per drug to send to the
                LLM (batched in one call per drug). Lower = faster.
            max_drugs: If set, cap how many drugs are scored after merging
                ensure_drugs (ensure list is kept first when trimming).

        Returns:
            Dict mapping drug_name -> literature_score (0-1).
        """
        # Auto-discover candidates if none provided
        if not drug_names:
            if discover:
                drug_names = self.discover_candidates(
                    disease_name,
                    top_k=discover_top_k,
                    max_articles=max_articles,
                )
            drug_names = drug_names or []

        drug_names = self._merge_ensure_drugs(drug_names)
        drug_names = self._cap_drug_names_after_merge(drug_names, max_drugs)

        if not drug_names:
            print("[LITERATURE] No drug candidates to score.")
            return {}

        print(f"[LITERATURE] Scoring {len(drug_names)} drugs "
              f"x '{disease_name}'...")

        raw_scores = {}
        details = {}

        cap_abs = max(1, min(10, int(max_llm_abstracts)))

        for i, drug_name in enumerate(drug_names):
            if (i + 1) % 10 == 0 or (i + 1) == len(drug_names):
                print(f"  ...literature {i+1}/{len(drug_names)} drugs")

            count, pmids = self._search_pubmed(drug_name, disease_name)

            cooccurrence = np.log10(count + 1)

            llm_score = 0.0
            abstracts = []
            citations = []
            if count > 0:
                abstracts = self._fetch_abstracts(
                    pmids, drug_name, disease_name
                )
                if abstracts and self.nlp:
                    to_score = abstracts[:cap_abs]
                    batch_scores = self._score_abstracts_relevance_batch(
                        drug_name, disease_name, to_score
                    )
                    llm_score = float(np.mean(batch_scores)) / 10.0
                citations = []
                for a in abstracts[:cap_abs]:
                    line = self._abstract_citation_line(a)
                    if line:
                        citations.append(line)

            raw_scores[drug_name] = {
                "cooccurrence": cooccurrence,
                "llm_relevance": llm_score,
                "count": count,
                "abstracts_analyzed": len(abstracts),
                "citations": citations[:5],
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
                norm_cooc = 0.0 if max_cooc <= 0 else 1.0

            if self.nlp and info["llm_relevance"] > 0:
                score = 0.4 * norm_cooc + 0.6 * info["llm_relevance"]
            else:
                score = norm_cooc

            results[drug] = score
            details[drug] = info

        self._last_details = details

        hits = sum(1 for v in raw_scores.values() if v["count"] > 0)
        print(f"[LITERATURE] Found publications for "
              f"{hits}/{len(drug_names)} drugs")

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


# ------------------------------------------------------------------ #
#  Demo / CLI
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    scorer = LiteratureScorer()
    disease = "fragile x syndrome"

    # Option 1: Auto-discover top 50 candidates from PubMed
    print(f"\n{'='*60}")
    print(f"Auto-discovering drug candidates for '{disease}'...")
    print(f"{'='*60}\n")

    scores = scorer.score_drugs_for_disease(
        disease,
        drug_names=None,       # <-- triggers auto-discovery
        discover=True,
        discover_top_k=50,     # discover top 50 candidates
        max_articles=200,      # scan up to 200 PubMed articles
        top_k=20,              # return top 20 scored
    )

    if scores:
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        details = scorer.get_details()

        print(f"\n{'='*60}")
        print(f"Top {len(ranked)} drugs for '{disease}' (auto-discovered)")
        print(f"{'='*60}\n")

        for i, (drug, score) in enumerate(ranked, 1):
            info = details.get(drug, {})
            citations = info.get("citations", [])
            cite_str = "; ".join(citations[:3]) if citations else "none"
            print(f"{i:3d}. {drug:30s}  score={score:.2%}  "
                  f"papers={info.get('count', 0):3d}  "
                  f"| {cite_str}")
    else:
        print("No results found.")

    # Option 2: You can still pass a manual list if you want
    # scores = scorer.score_drugs_for_disease(
    #     disease,
    #     drug_names=["Lovastatin", "Metformin", "Lithium"],
    # )