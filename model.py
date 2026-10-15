"""Data and model engine for ITViec Career Match & Review Sentiment AI (Project 2)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
MODELS = ASSETS / "models"


def clean_text(value: object) -> str:
    """Preprocess text matching Project 2 training pipeline: strip urls, tags, extra spaces."""
    if pd.isna(value):
        return ""
    text = re.sub(r"https?://\S+|<[^>]+>", " ", str(value))
    return re.sub(r"\s+", " ", text).strip().lower()


def escape_csv_cell(value: object) -> object:
    """Prevent spreadsheet formula injection when exporting CSV."""
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def sanitize_dataframe_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of DataFrame with text cells safely escaped for CSV export."""
    safe_df = df.copy()
    for col in safe_df.select_dtypes(include=["object", "string"]).columns:
        safe_df[col] = safe_df[col].map(escape_csv_cell)
    return safe_df


class ITViecEngine:
    """Encapsulates Recommender, Company Metadata, Reviews Database and Sentiment Classifier."""

    def __init__(self) -> None:
        # 1. Load full company metadata
        parquet_path = ASSETS / "companies.parquet"
        if parquet_path.exists():
            self.companies = pd.read_parquet(parquet_path)
        else:
            excel_path = ROOT.parent / "Project 2" / "data" / "Overview_Companies.xlsx"
            self.companies = pd.read_excel(excel_path).drop_duplicates("id").sort_values("id").reset_index(drop=True)

        # 2. Load Reviews dataset
        rev_parquet = ASSETS / "reviews.parquet"
        if rev_parquet.exists():
            self.reviews = pd.read_parquet(rev_parquet)
        else:
            rev_excel = ROOT.parent / "Project 2" / "data" / "Reviews.xlsx"
            if rev_excel.exists():
                self.reviews = pd.read_excel(rev_excel)
            else:
                self.reviews = pd.DataFrame()

        # 3. Load Recommender model
        rec_path = MODELS / "recommender.joblib"
        if not rec_path.exists():
            raise FileNotFoundError(f"Missing recommender model at {rec_path}")
        self.recommender = joblib.load(rec_path)
        self.rec_matrix = self.recommender["matrix"]
        self.rec_vectorizer = self.recommender["vectorizer"]

        # 4. Load Classifier model
        clf_path = MODELS / "classifier.joblib"
        if not clf_path.exists():
            raise FileNotFoundError(f"Missing classifier model at {clf_path}")
        self.classifier = joblib.load(clf_path)
        self.clf_model = self.classifier["model"]
        self.clf_tfidf = self.classifier["tfidf"]
        self.numeric_cols = self.classifier["numeric"]
        self.numeric_median = self.classifier["numeric_median"]

        # 5. Load JSON benchmarks
        self.classification_metrics = json.loads((ASSETS / "classification.json").read_text(encoding="utf-8"))
        self.recommendation_metrics = json.loads((ASSETS / "recommendation.json").read_text(encoding="utf-8"))
        self.eda_metrics = json.loads((ASSETS / "eda.json").read_text(encoding="utf-8"))

        # Precompute company search index
        self.company_names = self.companies["Company Name"].tolist()
        self.company_ids = [int(cid) for cid in self.companies["id"].tolist()]
        self.id_to_index = {cid: idx for idx, cid in enumerate(self.company_ids)}
        self.id_to_name = {cid: str(name) for cid, name in zip(self.company_ids, self.company_names)}

        # Precompute category filter options
        self.industries = sorted([
            str(x).strip() for x in self.companies["Company industry"].dropna().unique() if str(x).strip()
        ])
        self.company_types = sorted([
            str(x).strip() for x in self.companies["Company Type"].dropna().unique() if str(x).strip()
        ])
        self.company_sizes = sorted([
            str(x).strip() for x in self.companies["Company size"].dropna().unique() if str(x).strip()
        ])
        self.countries = sorted([
            str(x).strip() for x in self.companies["Country"].dropna().unique() if str(x).strip()
        ])

        # Precompute company review stats
        self.review_stats: dict[int, dict[str, Any]] = {}
        if not self.reviews.empty:
            rev_copy = self.reviews.copy()
            rev_copy["is_rec"] = rev_copy["Recommend?"].astype(str).str.strip().str.capitalize() == "Yes"
            grouped = rev_copy.groupby("id")
            for cid, grp in grouped:
                n_rev = len(grp)
                avg_r = float(grp["Rating"].mean())
                pct_rec = float(grp["is_rec"].mean() * 100)
                self.review_stats[int(cid)] = {
                    "n_reviews": n_rev,
                    "avg_rating": round(avg_r, 2),
                    "pct_recommend": round(pct_rec, 1),
                    "avg_salary": round(float(grp["Salary & benefits"].mean()), 2),
                    "avg_training": round(float(grp["Training & learning"].mean()), 2),
                    "avg_management": round(float(grp["Management cares about me"].mean()), 2),
                    "avg_culture": round(float(grp["Culture & fun"].mean()), 2),
                    "avg_office": round(float(grp["Office & workspace"].mean()), 2),
                }

    def get_company_display_name(self, cid: int | None) -> str:
        """Format company ID into clean human-readable name with ID badge."""
        if cid is None:
            return "(Không chọn)"
        name = self.id_to_name.get(int(cid), f"Company #{cid}")
        return f"{name} (ID: {cid})"

    def get_company_by_id(self, company_id: int) -> dict[str, Any] | None:
        """Retrieve full details of a company by its ID enriched with review aggregates."""
        matches = self.companies[self.companies["id"] == company_id]
        if matches.empty:
            return None
        data = matches.iloc[0].to_dict()
        stats = self.review_stats.get(company_id)
        if stats:
            data.update(stats)
        else:
            data.update({
                "n_reviews": 0,
                "avg_rating": None,
                "pct_recommend": None,
                "avg_salary": None,
                "avg_training": None,
                "avg_management": None,
                "avg_culture": None,
                "avg_office": None,
            })
        return data

    def get_company_reviews(self, company_id: int) -> pd.DataFrame:
        """Retrieve all reviews for a company sorted by recent first."""
        if self.reviews.empty:
            return pd.DataFrame()
        matches = self.reviews[self.reviews["id"] == company_id].copy()
        return matches.reset_index(drop=True)

    def get_top_rated_companies(self, min_reviews: int = 3, top_n: int = 10) -> list[dict[str, Any]]:
        """Return Top-N companies with highest ratings and positive reviews."""
        if not self.review_stats:
            return []
        items = []
        for cid, stats in self.review_stats.items():
            if stats["n_reviews"] >= min_reviews:
                comp = self.get_company_by_id(cid)
                if comp:
                    items.append(comp)
        items.sort(
            key=lambda x: (
                x.get("avg_rating") or 0.0,
                x.get("pct_recommend") or 0.0,
                x.get("n_reviews") or 0,
            ),
            reverse=True,
        )
        return items[:top_n]

    def recommend_by_company(
        self,
        company_id: int,
        top_k: int = 5,
        industry_filter: str | None = None,
        type_filter: str | None = None,
        size_filter: str | None = None,
        country_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Recommend Top-K companies based on cosine similarity with optional metadata filters."""
        if company_id not in self.id_to_index:
            return []
        query_idx = self.id_to_index[company_id]
        similarities = (self.rec_matrix[query_idx] @ self.rec_matrix.T).toarray().ravel()
        similarities[query_idx] = -1.0  # exclude the target company itself

        # Apply multi-attribute filters
        for idx in range(len(self.companies)):
            if similarities[idx] < 0:
                continue
            row = self.companies.iloc[idx]
            if industry_filter and industry_filter != "Tất cả" and str(row.get("Company industry", "")).strip() != industry_filter:
                similarities[idx] = -1.0
            elif type_filter and type_filter != "Tất cả" and str(row.get("Company Type", "")).strip() != type_filter:
                similarities[idx] = -1.0
            elif size_filter and size_filter != "Tất cả" and str(row.get("Company size", "")).strip() != size_filter:
                similarities[idx] = -1.0
            elif country_filter and country_filter != "Tất cả" and str(row.get("Country", "")).strip() != country_filter:
                similarities[idx] = -1.0

        valid_indices = np.where(similarities > 0)[0]
        if len(valid_indices) == 0:
            return []

        ranked_indices = valid_indices[np.argsort(-similarities[valid_indices], kind="stable")][:top_k]
        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            comp_row = self.companies.iloc[idx].to_dict()
            cid = int(comp_row["id"])
            stats = self.review_stats.get(cid, {})
            comp_row.update(stats)
            comp_row["rank"] = rank
            comp_row["similarity_score"] = float(similarities[idx])
            comp_row["similarity_pct"] = round(float(similarities[idx]) * 100, 1)
            results.append(comp_row)
        return results

    def recommend_by_candidate_profile(
        self,
        skills_query: str,
        industry_filter: str | None = None,
        type_filter: str | None = None,
        reference_company_id: int | None = None,
        top_k: int = 5,
        min_similarity_pct: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Recommend Top-K companies based on candidate skills, desired industry, type, and optional benchmark company."""
        combined_parts = [skills_query]
        if reference_company_id and reference_company_id in self.id_to_index:
            ref_comp = self.get_company_by_id(reference_company_id)
            if ref_comp:
                combined_parts.append(str(ref_comp.get("Our key skills", "")))
                combined_parts.append(str(ref_comp.get("Why you'll love working here", "")))
                combined_parts.append(str(ref_comp.get("Company overview", ""))[:300])

        if industry_filter and industry_filter != "Tất cả":
            combined_parts.append(industry_filter)
        if type_filter and type_filter != "Tất cả":
            combined_parts.append(type_filter)

        cleaned = clean_text(" ".join(combined_parts))
        if not cleaned:
            return []

        query_vec = self.rec_vectorizer.transform([cleaned])
        similarities = (query_vec @ self.rec_matrix.T).toarray().ravel()

        for idx in range(len(self.companies)):
            if similarities[idx] <= 0:
                continue
            row = self.companies.iloc[idx]
            if industry_filter and industry_filter != "Tất cả" and str(row.get("Company industry", "")).strip() != industry_filter:
                similarities[idx] = -1.0
            elif type_filter and type_filter != "Tất cả" and str(row.get("Company Type", "")).strip() != type_filter:
                similarities[idx] = -1.0

        valid_indices = np.where(similarities > 0)[0]
        if len(valid_indices) == 0:
            return []

        ranked_indices = valid_indices[np.argsort(-similarities[valid_indices], kind="stable")]
        results = []
        rank = 1
        for idx in ranked_indices:
            score = float(similarities[idx])
            sim_pct = round(score * 100, 1)
            if sim_pct < min_similarity_pct:
                continue
            comp_row = self.companies.iloc[idx].to_dict()
            cid = int(comp_row["id"])
            stats = self.review_stats.get(cid, {})
            comp_row.update(stats)
            comp_row["rank"] = rank
            comp_row["similarity_score"] = score
            comp_row["similarity_pct"] = sim_pct
            results.append(comp_row)
            rank += 1
            if len(results) >= top_k:
                break
        return results

    def get_gensim_comparison_for_company(self, company_name: str) -> dict[str, Any] | None:
        """Find precomputed comparison example between Scikit-learn and Gensim if available."""
        examples = self.recommendation_metrics.get("examples", [])
        for ex in examples:
            if ex.get("company", "").strip().lower() == company_name.strip().lower():
                return ex
        return None

    def predict_single_review(
        self,
        title: str,
        what_liked: str,
        suggestions: str,
        ratings: dict[str, float | int],
        company_id: int | None = None,
    ) -> dict[str, Any]:
        """Predict whether a review recommends the company (Yes vs No) with confidence and diagnostic."""
        combined_text = " ".join([str(title or ""), str(what_liked or ""), str(suggestions or "")])
        document = clean_text(combined_text)

        text_vec = self.clf_tfidf.transform([document])

        # Fill numeric ratings with medians if missing, normalize to [0, 1] (/ 5.0)
        num_vals = []
        for col in self.numeric_cols:
            val = ratings.get(col, self.numeric_median.get(col, 4.0))
            if val is None or np.isnan(val):
                val = float(self.numeric_median.get(col, 4.0))
            num_vals.append(float(val) / 5.0)

        num_vec = np.array([num_vals])
        features = hstack([text_vec, num_vec], format="csr")

        # In model: 0 = Yes (Recommend), 1 = No (Not Recommend)
        prediction = int(self.clf_model.predict(features)[0])
        probas = self.clf_model.predict_proba(features)[0]
        proba_yes = float(probas[0])
        proba_no = float(probas[1])

        label = "Yes" if prediction == 0 else "No"
        confidence = proba_yes if label == "Yes" else proba_no

        # Calculate average rating
        actual_ratings = {col: ratings.get(col, self.numeric_median.get(col, 4.0)) for col in self.numeric_cols}
        avg_score = float(np.mean(list(actual_ratings.values())))

        # Company context if provided
        company_info = None
        if company_id:
            company_info = self.get_company_by_id(company_id)

        return {
            "prediction": label,
            "is_recommend": (label == "Yes"),
            "confidence": confidence,
            "confidence_pct": round(confidence * 100, 1),
            "proba_yes": proba_yes,
            "proba_no": proba_no,
            "avg_score": round(avg_score, 2),
            "ratings": actual_ratings,
            "median_benchmark": self.numeric_median.to_dict(),
            "company_info": company_info,
        }

    def predict_batch_reviews(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any] | None]:
        """Batch predict reviews from uploaded DataFrame (CSV or Excel)."""
        work_df = df.copy()

        # Map text columns flexibly
        title_col = next((c for c in work_df.columns if str(c).strip().lower() in ["title", "tieu_de", "tiêu đề"]), None)
        liked_col = next((c for c in work_df.columns if str(c).strip().lower() in ["what i liked", "thich", "điểm thích", "liked"]), None)
        sugg_col = next((c for c in work_df.columns if str(c).strip().lower() in ["suggestions for improvement", "gop_y", "đề xuất", "cải thiện", "suggestions"]), None)

        docs = []
        for _, row in work_df.iterrows():
            parts = []
            if title_col and pd.notna(row[title_col]):
                parts.append(str(row[title_col]))
            if liked_col and pd.notna(row[liked_col]):
                parts.append(str(row[liked_col]))
            if sugg_col and pd.notna(row[sugg_col]):
                parts.append(str(row[sugg_col]))
            docs.append(clean_text(" ".join(parts)))

        text_vecs = self.clf_tfidf.transform(docs)

        # Map numeric columns
        num_rows = []
        for _, row in work_df.iterrows():
            row_nums = []
            for col in self.numeric_cols:
                matched_col = next((c for c in work_df.columns if str(c).strip().lower() == col.lower()), None)
                if matched_col and pd.notna(row[matched_col]):
                    try:
                        v = float(row[matched_col])
                    except (ValueError, TypeError):
                        v = float(self.numeric_median.get(col, 4.0))
                else:
                    v = float(self.numeric_median.get(col, 4.0))
                row_nums.append(v / 5.0)
            num_rows.append(row_nums)

        num_vecs = np.array(num_rows)
        features = hstack([text_vecs, num_vecs], format="csr")

        preds = self.clf_model.predict(features)
        probas = self.clf_model.predict_proba(features)

        result_df = work_df.copy()
        result_df["du_doan_recommend"] = ["Yes" if p == 0 else "No" for p in preds]
        result_df["xac_suat_tin_cay"] = [round(float(pr[p]) * 100, 1) for p, pr in zip(preds, probas)]
        result_df["xac_suat_yes"] = [round(float(pr[0]) * 100, 1) for pr in probas]
        result_df["xac_suat_no"] = [round(float(pr[1]) * 100, 1) for pr in probas]

        # Calculate evaluation metrics if actual label column exists
        actual_col = next((c for c in work_df.columns if str(c).strip().lower() in ["recommend?", "recommend", "nhãn", "label"]), None)
        evaluation = None
        if actual_col:
            valid_mask = work_df[actual_col].astype(str).str.strip().str.capitalize().isin(["Yes", "No"])
            if valid_mask.sum() > 0:
                y_true = (work_df.loc[valid_mask, actual_col].astype(str).str.strip().str.capitalize() == "No").astype(int)
                y_pred = preds[valid_mask]

                cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
                acc = float(accuracy_score(y_true, y_pred))
                p_no = float(precision_score(y_true, y_pred, zero_division=0))
                r_no = float(recall_score(y_true, y_pred, zero_division=0))
                f1_n = float(f1_score(y_true, y_pred, zero_division=0))

                evaluation = {
                    "total_evaluated": int(valid_mask.sum()),
                    "accuracy": acc,
                    "precision_no": p_no,
                    "recall_no": r_no,
                    "f1_no": f1_n,
                    "confusion_matrix": cm,
                }

        return result_df, evaluation
