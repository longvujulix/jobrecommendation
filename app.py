"""Streamlit GUI for Project 2: ITViec Career Match & Review Sentiment AI.

Full-stack application for:
1. Content-Based Company Recommendation (Multi-attribute Filters & Candidate Profile Matching)
2. Review Sentiment Classification (Predict 'Recommend?' Yes vs No with Company Context & Expectation Matching)
3. Company Profile & Full Reviews Explorer (Deep-dive, Employee Ratings & Head-to-Head Comparison)
4. Batch Review Processing (CSV/Excel upload, instant prediction, metrics & safe export)
5. Model Performance & Technical Benchmarks (Sklearn vs PySpark, imbalanced data analysis, visual plots)
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from model import ITViecEngine, ROOT, sanitize_dataframe_for_csv

# Configure Streamlit page layout and title
st.set_page_config(
    page_title="ITViec Hub · Career Match & Review AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom modern stylesheet
style_path = ROOT / "style.css"
if style_path.exists():
    st.markdown(f"<style>{style_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_resource
def get_engine() -> ITViecEngine:
    """Load engine and models once in memory."""
    return ITViecEngine()


try:
    engine = get_engine()
except Exception as exc:
    st.error(f"❌ Không thể nạp mô hình hoặc dữ liệu: {exc}. Vui lòng kiểm tra thư mục assets/.")
    st.stop()


def download_csv_bytes(df: pd.DataFrame) -> bytes:
    """Return UTF-8-sig CSV bytes with formula-injection guard."""
    safe = sanitize_dataframe_for_csv(df)
    return safe.to_csv(index=False).encode("utf-8-sig")


# ==========================================
# SIDEBAR NAVIGATION & AUTHOR PROFILE
# ==========================================
PAGES = [
    "🌐 Tổng quan",
    "🏢 Gợi ý công ty",
    "✍️ Đánh giá review",
    "📑 Hồ sơ & Review công ty",
    "📁 Dự đoán hàng loạt",
    "🔬 Báo cáo mô hình",
]

# Support cross-page navigation via session_state
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = PAGES[0]

with st.sidebar:
    st.markdown(
        """
        <div class="brand-container">
            <div class="brand-icon-box">💼</div>
            <div>
                <div class="brand-title">ITVIEC<span>HUB</span></div>
                <div class="brand-subtitle">Project 2 · Recommender & Sentiment</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio("Không gian làm việc", PAGES, key="nav_page")
    st.divider()

    st.markdown(
        """
        <div class="status-badge">
            <span class="pulse-dot"></span>
            <span>TF-IDF + Random Forest · 150 cây</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("🏢 **478 công ty IT** · 8.417 review từ 180 công ty.")
    st.caption("🔒 Dữ liệu upload chỉ xử lý trong phiên; không lưu trữ trên máy chủ.")

    st.divider()
    st.markdown(
        """
        <div class="author-card">
            <div class="author-header">
                <div class="author-avatar">QP</div>
                <div>
                    <div class="author-title">Học viên thực hiện</div>
                    <div class="author-name">Đặng Quốc Phong</div>
                </div>
            </div>
            <div class="author-email">✉️ <a href="mailto:phong.dang@samedtech.edu.vn" style="color:#e11d48; text-decoration:none; font-weight:600;">phong.dang@samedtech.edu.vn</a></div>
            <div class="author-role">🎓 Đồ án tốt nghiệp Data Science<br>Trung tâm Tin học · ĐH KHTN TP.HCM</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# PAGE 1: TỔNG QUAN (OVERVIEW & ECOSYSTEM)
# ==========================================
if page == "🌐 Tổng quan":
    st.markdown(
        """
        <div class="hero-box">
            <span class="hero-eyebrow">HỆ THỐNG ĐỒ ÁN DATA SCIENCE · TOPIC 2</span>
            <div class="hero-title">Gợi ý công ty phù hợp & <span>Dự đoán Review</span></div>
            <div class="hero-desc">
                Nền tảng trí tuệ nhân tạo toàn diện trên dữ liệu ITViec: Khám phá các công ty tương đồng về công nghệ, văn hóa qua
                <b>Content-Based Filtering (Cosine Similarity)</b> và dự đoán mức độ hài lòng của nhân viên (<b>Recommend: Yes/No</b>)
                bằng mô hình máy học tối ưu cho dữ liệu mất cân bằng.
            </div>
            <div class="hero-pills">
                <span class="hero-pill">🏢 478 Công ty CNTT</span>
                <span class="hero-pill">💬 8.417 Đánh giá thực tế</span>
                <span class="hero-pill">🎯 9.891 Chiều đặc trưng TF-IDF</span>
                <span class="hero-pill">🌲 Random Forest (Sklearn) & GBT (PySpark)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top metrics ribbon
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng công ty IT", f"{len(engine.companies):,}")
    c2.metric("Số lượt Review", f"{engine.eda_metrics['reviews']:,}")
    c3.metric("F1-score (Lớp No)", "0.669 (GBT) / 0.666 (SVC)", help="F1 trên lớp thiểu số No (chỉ 12.3% dữ liệu)")
    c4.metric("Đặc trưng TF-IDF", f"{engine.recommendation_metrics['sklearn_features']:,}")

    st.write("")
    st.subheader("Bạn muốn bắt đầu từ đâu?")

    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        with st.container(border=True):
            st.caption("01 / WORKSPACE")
            st.markdown("### 🏢 Gợi ý công ty")
            st.write(
                "Tìm kiếm doanh nghiệp tương đồng với bộ lọc đa tiêu chí (Lĩnh vực, Loại hình, Kỹ năng, Công ty mẫu)."
            )
            st.button(
                "Mở Recommender →",
                key="btn_open_rec",
                type="primary",
                on_click=lambda: st.session_state.update(nav_page="🏢 Gợi ý công ty"),
            )

    with col_b:
        with st.container(border=True):
            st.caption("02 / WORKSPACE")
            st.markdown("### ✍️ Đánh giá review")
            st.write(
                "Chọn công ty mục tiêu, chọn kỳ vọng và viết nhận xét để mô hình dự đoán nhãn Recommend có bối cảnh."
            )
            st.button(
                "Mở Classifier →",
                key="btn_open_clf",
                on_click=lambda: st.session_state.update(nav_page="✍️ Đánh giá review"),
            )

    with col_c:
        with st.container(border=True):
            st.caption("03 / WORKSPACE")
            st.markdown("### 📑 Hồ sơ & Review")
            st.write(
                "Tra cứu chi tiết từng công ty, biểu đồ đối chiếu 6 tiêu chí và đọc danh sách review thực tế từ nhân viên."
            )
            st.button(
                "Xem Hồ sơ công ty →",
                key="btn_open_profile",
                on_click=lambda: st.session_state.update(nav_page="📑 Hồ sơ & Review công ty"),
            )

    with col_d:
        with st.container(border=True):
            st.caption("04 / WORKSPACE")
            st.markdown("### 📁 Dự đoán hàng loạt")
            st.write(
                "Tải lên tệp CSV/Excel nhiều bài review để dự đoán nhãn tự động, xem ma trận nhầm lẫn và tải báo cáo."
            )
            st.button(
                "Mở Batch Processor →",
                key="btn_open_batch",
                on_click=lambda: st.session_state.update(nav_page="📁 Dự đoán hàng loạt"),
            )

    st.write("")
    st.divider()

    # BONUS: Top Rated Companies Section
    st.subheader("🏆 Bảng vàng doanh nghiệp: Top công ty có review cao & tích cực nhất")
    st.caption("Các công ty đạt điểm đánh giá cao nhất (từ 3 bài review thực tế trở lên), 100% nhân viên khuyên chọn làm việc.")

    top_companies = engine.get_top_rated_companies(min_reviews=3, top_n=6)
    if top_companies:
        top_cols = st.columns(3)
        for i, comp in enumerate(top_companies):
            col_idx = i % 3
            with top_cols[col_idx]:
                with st.container(border=True):
                    st.markdown(
                        f"""
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                            <span class="top-comp-badge">🥇 TOP #{i + 1} ITVIEC</span>
                            <span class="star-rating">⭐ {comp.get('avg_rating', 5.0):.2f} / 5.0</span>
                        </div>
                        <div style="font-family:'Outfit', sans-serif; font-size:1.15rem; font-weight:800; color:#0f172a; margin-bottom:4px;">
                            {comp['Company Name']}
                        </div>
                        <div style="font-size:0.8rem; color:#64748b; margin-bottom:8px;">
                            🏢 {comp.get('Company industry', 'IT')} · 💼 {comp.get('Company Type', 'IT')}
                        </div>
                        <div style="background:#ecfdf5; border:1px solid #a7f3d0; border-radius:8px; padding:6px 10px; font-size:0.82rem; color:#065f46; font-weight:600; margin-bottom:10px;">
                            👍 {comp.get('pct_recommend', 100.0):.0f}% khuyên chọn ({comp.get('n_reviews', 0)} bài review)
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if st.button("Xem hồ sơ & Đọc review →", key=f"btn_top_view_{comp['id']}", type="secondary"):
                        st.session_state["target_company_view"] = comp["id"]
                        st.session_state["nav_page"] = "📑 Hồ sơ & Review công ty"
                        st.rerun()

    st.write("")
    st.divider()

    # Visual data highlights
    st.subheader("📊 Bức tranh dữ liệu thực nghiệm (EDA Insights)")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.write("**Phân bố nhãn Recommend? (Bài toán lệch nhãn Imbalance)**")
        labels_dict = engine.eda_metrics.get("recommend_labels", {"Yes": 7385, "No": 1032})
        df_labels = pd.DataFrame(
            {"Nhãn": ["Yes (Khuyên chọn)", "No (Không khuyên)"], "Số lượng review": [labels_dict.get("Yes", 0), labels_dict.get("No", 0)]}
        ).set_index("Nhãn")
        st.bar_chart(df_labels, color="#e11d48")
        st.caption("💡 Lớp 'No' chiếm 12.3%, do đó F1-score trên lớp No được dùng làm chỉ số tối thượng để chọn mô hình.")

    with chart_col2:
        st.write("**Phân bố Rating tổng thể (1 - 5 sao)**")
        rating_cross = engine.eda_metrics.get("rating_by_recommend", {})
        ratings_data = []
        for r_score in ["1", "2", "3", "4", "5"]:
            item = rating_cross.get(r_score, {})
            ratings_data.append({"Rating": f"{r_score} sao", "Yes": item.get("Yes", 0), "No": item.get("No", 0)})
        df_rating = pd.DataFrame(ratings_data).set_index("Rating")
        st.bar_chart(df_rating, color=["#10b981", "#e11d48"])
        st.caption("💡 Điểm 1-2 sao phần lớn là 'No', trong khi 4-5 sao chiếm ưu thế áp đảo cho 'Yes'.")


# ==========================================
# PAGE 2: GỢI Ý CÔNG TY (RECOMMENDER SYSTEM)
# ==========================================
elif page == "🏢 Gợi ý công ty":
    st.title("🏢 Hệ thống gợi ý công ty (Content-Based Filtering)")
    st.write(
        "Tìm kiếm và đối chiếu các công ty công nghệ dựa trên độ tương đồng nội dung văn bản (Company overview, Our key skills, Why you'll love working here) kết hợp bộ lọc đa tiêu chí."
    )

    tab_by_company, tab_by_skills = st.tabs(["🔍 Gợi ý theo công ty mục tiêu", "🎯 Tìm theo kỹ năng ứng viên"])

    # --- TAB 1: GỢI Ý THEO CÔNG TY ĐÃ CHỌN ---
    with tab_by_company:
        st.subheader("1. Chọn công ty để tìm các doanh nghiệp tương tự")

        col_sel, col_k = st.columns([3, 1])
        with col_sel:
            selected_id = st.selectbox(
                "Chọn công ty từ cơ sở dữ liệu ITViec",
                options=engine.company_ids,
                format_func=engine.get_company_display_name,
                key="select_company_target",
            )
        with col_k:
            top_k = st.slider("Số lượng gợi ý (Top K)", min_value=1, max_value=15, value=5, key="top_k_rec")

        # Display Selected Company Profile
        curr_comp = engine.get_company_by_id(selected_id)
        if curr_comp:
            with st.container(border=True):
                st.caption("CÔNG TY MỤC TIÊU ĐÃ CHỌN")
                st.markdown(f"### {curr_comp.get('Company Name')}")

                # Badges row
                b1, b2, b3, b4 = st.columns(4)
                b1.markdown(f"**Loại hình:** {curr_comp.get('Company Type', 'N/A')}")
                b2.markdown(f"**Ngành nghề:** {curr_comp.get('Company industry', 'N/A')}")
                b3.markdown(f"**Quy mô:** {curr_comp.get('Company size', 'N/A')}")
                b4.markdown(f"**Quốc gia:** {curr_comp.get('Country', 'N/A')}")

                if curr_comp.get("avg_rating"):
                    st.markdown(f"⭐ **Điểm nhân viên đánh giá:** `{curr_comp['avg_rating']:.2f} / 5.0` ({curr_comp['n_reviews']} reviews · {curr_comp['pct_recommend']:.0f}% Recommend)")

                # Key skills & Why love working here
                skills_val = curr_comp.get("Our key skills")
                if pd.notna(skills_val) and str(skills_val).strip():
                    st.markdown(f"**Kỹ năng chủ đạo:** `{skills_val}`")

                why_val = curr_comp.get("Why you'll love working here")
                if pd.notna(why_val) and str(why_val).strip():
                    with st.expander("✨ Lý do nhân viên yêu thích làm việc tại đây", expanded=False):
                        st.write(why_val)

                overview_val = curr_comp.get("Company overview")
                if pd.notna(overview_val) and str(overview_val).strip():
                    with st.expander("📄 Tổng quan công ty (Overview)", expanded=False):
                        st.write(str(overview_val)[:400] + ("..." if len(str(overview_val)) > 400 else ""))

                href_val = curr_comp.get("Href")
                if pd.notna(href_val) and str(href_val).strip():
                    st.markdown(f"[🔗 Xem trang công ty trên ITViec]({str(href_val).strip()})")

        st.write("")
        st.markdown("#### 🎯 Bộ lọc nâng cao (Lọc theo Lĩnh vực & Ngành nghề)")
        st.caption("Thu hẹp phạm vi gợi ý để chỉ nhận các công ty đúng lĩnh vực hoặc mô hình mong muốn:")

        flt_col1, flt_col2, flt_col3, flt_col4 = st.columns(4)
        with flt_col1:
            sel_industry = st.selectbox(
                "Lĩnh vực / Ngành nghề",
                ["Tất cả"] + engine.industries,
                key="filter_rec_industry",
            )
        with flt_col2:
            sel_type = st.selectbox(
                "Loại hình công ty",
                ["Tất cả"] + engine.company_types,
                key="filter_rec_type",
            )
        with flt_col3:
            sel_size = st.selectbox(
                "Quy mô nhân sự",
                ["Tất cả"] + engine.company_sizes,
                key="filter_rec_size",
            )
        with flt_col4:
            sel_country = st.selectbox(
                "Quốc gia",
                ["Tất cả"] + engine.countries,
                key="filter_rec_country",
            )

        st.write("")
        st.subheader(f"Top {top_k} công ty tương đồng nhất")

        recs = engine.recommend_by_company(
            selected_id,
            top_k=top_k,
            industry_filter=sel_industry if sel_industry != "Tất cả" else None,
            type_filter=sel_type if sel_type != "Tất cả" else None,
            size_filter=sel_size if sel_size != "Tất cả" else None,
            country_filter=sel_country if sel_country != "Tất cả" else None,
        )

        if not recs:
            st.warning("Không tìm thấy công ty nào thỏa mãn cả độ tương đồng và các bộ lọc đã chọn. Hãy thử nới lỏng bộ lọc.")
        else:
            for rec_item in recs:
                st.markdown(
                    f"""
                    <div class="company-card">
                        <div class="company-header">
                            <div>
                                <div class="company-name">#{rec_item['rank']}. {rec_item['Company Name']}</div>
                                <div style="margin-top: 6px;">
                                    <span class="chip chip-industry">🏢 {rec_item.get('Company industry', 'IT')}</span>
                                    <span class="chip chip-type">💼 {rec_item.get('Company Type', 'IT')}</span>
                                    <span class="chip chip-size">👥 {rec_item.get('Company size', 'N/A')}</span>
                                    <span class="chip chip-country">🌏 {rec_item.get('Country', 'Vietnam')}</span>
                                    {'<span class="chip" style="background:#fefce8; color:#854d0e; font-weight:700;">⭐ ' + str(rec_item.get('avg_rating', '')) + ' / 5.0</span>' if rec_item.get('avg_rating') else ''}
                                </div>
                            </div>
                            <div class="similarity-badge">
                                🎯 {rec_item['similarity_pct']}% Tương đồng
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Expandable details
                with st.expander(f"Chi tiết #{rec_item['rank']}: {rec_item['Company Name']}", expanded=False):
                    if pd.notna(rec_item.get("Our key skills")):
                        st.markdown(f"**Kỹ năng:** `{rec_item['Our key skills']}`")
                    if pd.notna(rec_item.get("Why you'll love working here")):
                        why_desc = rec_item.get("Why you'll love working here")
                        st.markdown(f"**Lợi thế làm việc:** {why_desc}")
                    if pd.notna(rec_item.get("Company overview")):
                        st.markdown(f"**Mô tả:** {str(rec_item['Company overview'])[:300]}...")
                    if pd.notna(rec_item.get("Href")):
                        st.markdown(f"[🔗 Xem chi tiết trên ITViec]({rec_item['Href']})")

                    if st.button("📑 Xem toàn bộ review công ty này", key=f"btn_goto_profile_{rec_item['id']}"):
                        st.session_state["target_company_view"] = rec_item["id"]
                        st.session_state["nav_page"] = "📑 Hồ sơ & Review công ty"
                        st.rerun()

        # Technical check: Gensim comparison if exists
        gensim_comp = engine.get_gensim_comparison_for_company(curr_comp.get("Company Name", ""))
        if gensim_comp:
            st.write("")
            with st.expander("🔬 Đối chiếu kết quả giữa Scikit-Learn TF-IDF và Gensim Topic Model", expanded=True):
                st.caption(
                    "Theo yêu cầu đồ án: So sánh độ trùng lặp giữa 2 phương pháp biểu diễn vector không gian (Scikit-Learn TF-IDF vs Gensim)."
                )
                col_sk, col_gs = st.columns(2)
                with col_sk:
                    st.write("**Top-5 theo Scikit-Learn (TF-IDF):**")
                    df_sk = pd.DataFrame(gensim_comp.get("sklearn", []))
                    st.dataframe(df_sk, hide_index=True, width="stretch")
                with col_gs:
                    st.write("**Top-5 theo Gensim:**")
                    df_gs = pd.DataFrame(gensim_comp.get("gensim", []))
                    st.dataframe(df_gs, hide_index=True, width="stretch")

    # --- TAB 2: GỢI Ý THEO KỸ NĂNG & HỒ SƠ ỨNG VIÊN ---
    with tab_by_skills:
        st.subheader("2. Tìm công ty theo kỹ năng & hồ sơ ứng viên")
        st.write(
            "Nhập các từ khóa công nghệ, framework, kết hợp với lĩnh vực mong muốn hoặc chọn công ty hình mẫu tham chiếu để tìm các doanh nghiệp phù hợp nhất."
        )

        query_input = st.text_area(
            "Kỹ năng & Môi trường mong muốn",
            value="Python React microservices cloud AWS văn hóa năng động hybrid working chế độ tốt",
            height=100,
            key="candidate_query_input",
        )

        col_q_ind, col_q_typ, col_q_ref = st.columns(3)
        with col_q_ind:
            query_ind = st.selectbox(
                "Lĩnh vực mong muốn (Tùy chọn)",
                ["Tất cả"] + engine.industries,
                key="query_filter_industry",
            )
        with col_q_typ:
            query_typ = st.selectbox(
                "Loại hình công ty mong muốn (Tùy chọn)",
                ["Tất cả"] + engine.company_types,
                key="query_filter_type",
            )
        with col_q_ref:
            query_ref_id = st.selectbox(
                "Công ty hình mẫu tham chiếu (Tùy chọn)",
                options=[None] + engine.company_ids,
                format_func=engine.get_company_display_name,
                key="query_ref_company",
                help="Nếu bạn có 1 công ty rất thích, hệ thống sẽ kết hợp văn hóa/tech stack của công ty đó vào hồ sơ của bạn!",
            )

        col_q1, col_q2 = st.columns([1, 1])
        with col_q1:
            query_top_k = st.slider("Số lượng gợi ý (Top K)", 1, 15, 5, key="query_top_k")
        with col_q2:
            query_min_sim = st.slider("Độ tương đồng tối thiểu (%)", 0, 40, 5, key="query_min_sim")

        if st.button("Tìm công ty phù hợp 🚀", type="primary", key="btn_search_skills"):
            if not query_input.strip() and not query_ref_id:
                st.warning("Vui lòng nhập từ khóa kỹ năng hoặc chọn công ty tham chiếu.")
            else:
                with st.spinner("Đang tính toán vector không gian TF-IDF và trích xuất độ tương đồng..."):
                    q_recs = engine.recommend_by_candidate_profile(
                        skills_query=query_input,
                        industry_filter=query_ind if query_ind != "Tất cả" else None,
                        type_filter=query_typ if query_typ != "Tất cả" else None,
                        reference_company_id=query_ref_id,
                        top_k=query_top_k,
                        min_similarity_pct=float(query_min_sim),
                    )

                if not q_recs:
                    st.info("Không có công ty nào khớp với tiêu chuẩn đã chọn. Hãy thử giảm độ tương đồng tối thiểu hoặc chọn 'Tất cả' lĩnh vực.")
                else:
                    st.success(f"Đã tìm thấy {len(q_recs)} công ty phù hợp nhất với hồ sơ của bạn:")
                    for q_item in q_recs:
                        st.markdown(
                            f"""
                            <div class="company-card">
                                <div class="company-header">
                                    <div>
                                        <div class="company-name">#{q_item['rank']}. {q_item['Company Name']}</div>
                                        <div style="margin-top: 6px;">
                                            <span class="chip chip-industry">🏢 {q_item.get('Company industry', 'IT')}</span>
                                            <span class="chip chip-type">💼 {q_item.get('Company Type', 'IT')}</span>
                                            <span class="chip chip-size">👥 {q_item.get('Company size', 'N/A')}</span>
                                            {'<span class="chip" style="background:#fefce8; color:#854d0e; font-weight:700;">⭐ ' + str(q_item.get('avg_rating', '')) + ' / 5.0</span>' if q_item.get('avg_rating') else ''}
                                        </div>
                                    </div>
                                    <div class="similarity-badge">
                                        🎯 {q_item['similarity_pct']}% Điểm khớp
                                    </div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        with st.expander(f"Xem hồ sơ công ty #{q_item['rank']}: {q_item['Company Name']}"):
                            if pd.notna(q_item.get("Our key skills")):
                                st.markdown(f"**Key skills:** `{q_item['Our key skills']}`")
                            if pd.notna(q_item.get("Why you'll love working here")):
                                q_why_desc = q_item.get("Why you'll love working here")
                                st.markdown(f"**Why love:** {q_why_desc}")
                            if pd.notna(q_item.get("Company overview")):
                                st.markdown(f"**Overview:** {str(q_item['Company overview'])[:300]}...")
                            if st.button("📑 Xem hồ sơ & Toàn bộ review", key=f"btn_skill_view_{q_item['id']}"):
                                st.session_state["target_company_view"] = q_item["id"]
                                st.session_state["nav_page"] = "📑 Hồ sơ & Review công ty"
                                st.rerun()


# ==========================================
# PAGE 3: ĐÁNH GIÁ REVIEW (SINGLE REVIEW AI)
# ==========================================
elif page == "✍️ Đánh giá review":
    st.title("✍️ Dự đoán nhãn Review ('Recommend or Not')")
    st.write(
        "Quy trình 3 bước nghiệp vụ: Chọn công ty cần đánh giá, xác định kỳ vọng cá nhân, nhập nội dung nhận xét và chấm 6 tiêu chí để AI dự đoán liệu bạn có khuyên ứng viên khác gia nhập công ty đó hay không."
    )

    # STEP 1: CHỌN CÔNG TY MỤC TIÊU
    st.markdown("### Bước 1: Chọn công ty bạn muốn đánh giá")
    eval_comp_id = st.selectbox(
        "Tìm kiếm và chọn công ty trong hệ thống ITViec",
        options=engine.company_ids,
        format_func=engine.get_company_display_name,
        key="eval_select_company",
    )
    eval_comp_info = engine.get_company_by_id(eval_comp_id)

    if eval_comp_info:
        with st.container(border=True):
            st.caption("CÔNG TY ĐANG ĐƯỢC ĐÁNH GIÁ")
            st.markdown(f"#### 🏢 {eval_comp_info['Company Name']}")
            c_inf1, c_inf2, c_inf3, c_inf4 = st.columns(4)
            c_inf1.markdown(f"**Loại hình:** {eval_comp_info.get('Company Type', 'IT')}")
            c_inf2.markdown(f"**Ngành nghề:** {eval_comp_info.get('Company industry', 'IT')}")
            c_inf3.markdown(f"**Quy mô:** {eval_comp_info.get('Company size', 'N/A')}")
            c_inf4.markdown(f"**Quốc gia:** {eval_comp_info.get('Country', 'N/A')}")

            if eval_comp_info.get("avg_rating"):
                st.markdown(
                    f"📊 **Lịch sử đánh giá trên ITViec:** ⭐ **{eval_comp_info['avg_rating']:.2f} / 5.0** "
                    f"· **{eval_comp_info['pct_recommend']:.0f}%** nhân viên khuyên chọn ({eval_comp_info['n_reviews']} bài review trong cơ sở dữ liệu)."
                )

    # Preset templates for quick testing
    st.write("")
    st.caption("Thử nghiệm nhanh với mẫu đánh giá:")
    c_btn1, c_btn2, c_btn3 = st.columns([1.5, 1.5, 2])
    if c_btn1.button("👍 Mẫu review tích cực", key="btn_sample_pos"):
        st.session_state["review_title"] = "Môi trường năng động, đồng nghiệp thân thiện và lộ trình rõ ràng"
        st.session_state["review_liked"] = "Lương thưởng cạnh tranh, cơ hội học hỏi công nghệ mới, sếp rất biết lắng nghe và hỗ trợ."
        st.session_state["review_sugg"] = "Cần tăng thêm các hoạt động teambuilding gắn kết phòng ban."
        st.session_state["val_rating"] = 5
        st.session_state["val_salary"] = 5
        st.session_state["val_training"] = 4
        st.session_state["val_management"] = 5
        st.session_state["val_culture"] = 5
        st.session_state["val_office"] = 4

    if c_btn2.button("👎 Mẫu review không khuyến nghị", key="btn_sample_neg"):
        st.session_state["review_title"] = "Áp lực OT triền miên, chế độ đãi ngộ chưa tương xứng"
        st.session_state["review_liked"] = "Văn phòng nằm ở vị trí trung tâm, đi lại thuận tiện."
        st.session_state["review_sugg"] = "Cần tôn trọng thời gian cá nhân của nhân viên, trả lương OT công bằng và giảm tải áp lực quản lý vi mô."
        st.session_state["val_rating"] = 1
        st.session_state["val_salary"] = 2
        st.session_state["val_training"] = 1
        st.session_state["val_management"] = 1
        st.session_state["val_culture"] = 2
        st.session_state["val_office"] = 3

    # Load actual sample from this company if available
    comp_reviews = engine.get_company_reviews(eval_comp_id)
    if not comp_reviews.empty and c_btn3.button(f"📥 Nạp review thực tế của {eval_comp_info['Company Name'][:15]}...", key="btn_sample_actual"):
        first_rev = comp_reviews.iloc[0]
        st.session_state["review_title"] = str(first_rev.get("Title", ""))
        st.session_state["review_liked"] = str(first_rev.get("What I liked", ""))
        st.session_state["review_sugg"] = str(first_rev.get("Suggestions for improvement", ""))
        st.session_state["val_rating"] = int(first_rev.get("Rating", 4))
        st.session_state["val_salary"] = int(first_rev.get("Salary & benefits", 4))
        st.session_state["val_training"] = int(first_rev.get("Training & learning", 4))
        st.session_state["val_management"] = int(first_rev.get("Management cares about me", 4))
        st.session_state["val_culture"] = int(first_rev.get("Culture & fun", 4))
        st.session_state["val_office"] = int(first_rev.get("Office & workspace", 4))

    st.write("")
    st.markdown("### Bước 2: Chọn kỳ vọng của bạn & Viết nhận xét")

    candidate_expectation = st.selectbox(
        "Kỳ vọng ưu tiên số 1 của bạn đối với công ty này",
        [
            "💰 Lương thưởng & Chế độ phúc lợi cao",
            "📚 Môi trường đào tạo bài bản & Cơ hội học hỏi công nghệ mới",
            "⚖️ Work-Life Balance (Cân bằng cuộc sống, hạn chế áp lực OT)",
            "🤝 Ban quản lý có tầm nhìn & Tôn trọng, lắng nghe nhân viên",
            "🎉 Văn hóa công ty trẻ trung, hòa đồng, nhiều hoạt động gắn kết",
            "🏢 Văn phòng làm việc hiện đại, tiện nghi, thiết bị máy móc xịn",
        ],
        key="sel_candidate_expectation",
    )

    col_text, col_ratings = st.columns([1.2, 1])

    with col_text:
        st.subheader("2.1. Nội dung đánh giá (Text NLP)")
        r_title = st.text_input(
            "Tiêu đề bài review (Title)",
            value=st.session_state.get("review_title", "Công ty có môi trường làm việc tốt và đồng nghiệp trẻ"),
            key="input_review_title",
        )
        r_liked = st.text_area(
            "Điều bạn thích nhất (What I liked)",
            value=st.session_state.get("review_liked", "Chế độ đãi ngộ tốt, lương thưởng đúng hạn, sếp cởi mở hướng dẫn nhiệt tình."),
            height=100,
            key="input_review_liked",
        )
        r_sugg = st.text_area(
            "Đề xuất cải thiện (Suggestions for improvement)",
            value=st.session_state.get("review_sugg", "Quy trình dự án đôi khi còn chậm, cần tối ưu hóa công cụ quản lý task."),
            height=90,
            key="input_review_sugg",
        )

    with col_ratings:
        st.subheader("2.2. Điểm số 6 tiêu chí (Thang 1 - 5)")
        s_rating = st.slider("⭐ Điểm tổng quan (Rating)", 1, 5, st.session_state.get("val_rating", 4), key="sl_rating")
        s_salary = st.slider("💰 Lương & Phúc lợi (Salary & benefits)", 1, 5, st.session_state.get("val_salary", 4), key="sl_salary")
        s_train = st.slider("📚 Đào tạo & Học hỏi (Training & learning)", 1, 5, st.session_state.get("val_training", 4), key="sl_train")
        s_mgmt = st.slider("🤝 Ban quản lý quan tâm (Management cares)", 1, 5, st.session_state.get("val_management", 4), key="sl_mgmt")
        s_cult = st.slider("🎉 Văn hóa & Niềm vui (Culture & fun)", 1, 5, st.session_state.get("val_culture", 4), key="sl_cult")
        s_off = st.slider("🏢 Không gian làm việc (Office & workspace)", 1, 5, st.session_state.get("val_office", 4), key="sl_off")

    st.write("")
    st.markdown("### Bước 3: Dự đoán & Phân tích kết quả")

    if st.button("Dự đoán nhãn Recommend cho công ty này ⚡", type="primary", key="btn_predict_single"):
        ratings_dict = {
            "Rating": s_rating,
            "Salary & benefits": s_salary,
            "Training & learning": s_train,
            "Management cares about me": s_mgmt,
            "Culture & fun": s_cult,
            "Office & workspace": s_off,
        }

        with st.spinner(f"Đang phân tích đánh giá cho công ty {eval_comp_info['Company Name']}..."):
            pred_res = engine.predict_single_review(
                r_title, r_liked, r_sugg, ratings_dict, company_id=eval_comp_id
            )

        st.write("")
        st.subheader(f"Kết quả dự đoán đối với công ty: {eval_comp_info['Company Name']}")

        if pred_res["is_recommend"]:
            st.markdown(
                f"""
                <div class="prediction-box-yes">
                    <div class="prediction-title-yes">
                        👍 RECOMMEND (KHUYÊN GIA NHẬP {eval_comp_info['Company Name'].upper()})
                    </div>
                    <div style="margin-top: 10px; font-size: 1.05rem; color: #065f46; line-height:1.5;">
                        Mô hình AI dự đoán với trải nghiệm này, bạn <b>SẼ KHUYÊN ỨNG VIÊN KHÁC GIA NHẬP</b> công ty <b>{eval_comp_info['Company Name']}</b>.<br>
                        Độ tin cậy của mô hình: <b>{pred_res['confidence_pct']}%</b> (Xác suất Yes: {pred_res['proba_yes'] * 100:.1f}% · No: {pred_res['proba_no'] * 100:.1f}%)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="prediction-box-no">
                    <div class="prediction-title-no">
                        👎 NOT RECOMMEND (KHÔNG KHUYÊN GIA NHẬP {eval_comp_info['Company Name'].upper()})
                    </div>
                    <div style="margin-top: 10px; font-size: 1.05rem; color: #9f1239; line-height:1.5;">
                        Mô hình AI dự đoán với trải nghiệm này, bạn <b>SẼ KHÔNG KHUYÊN ỨNG VIÊN KHÁC GIA NHẬP</b> công ty <b>{eval_comp_info['Company Name']}</b>.<br>
                        Độ tin cậy của mô hình: <b>{pred_res['confidence_pct']}%</b> (Xác suất No: {pred_res['proba_no'] * 100:.1f}% · Yes: {pred_res['proba_yes'] * 100:.1f}%)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Contextual insights comparing to user's expectation & company history
        with st.container(border=True):
            st.markdown("#### 🔍 Nhận định đối chiếu giữa Kỳ vọng và Điểm đánh giá")
            st.write(f"• **Kỳ vọng đã chọn:** {candidate_expectation}")

            # Check if user's expectation criterion got high or low score
            exp_key_map = {
                "Lương thưởng": s_salary,
                "Đào tạo": s_train,
                "Work-Life Balance": s_mgmt,
                "Ban quản lý": s_mgmt,
                "Văn hóa": s_cult,
                "Văn phòng": s_off,
            }
            for kw, score_val in exp_key_map.items():
                if kw in candidate_expectation:
                    if score_val >= 4:
                        st.success(f"✓ Tiêu chí quan trọng nhất của bạn ({kw}) đạt mức chấm cao ({score_val}/5 sao), hoàn toàn phù hợp với kết luận Recommend tích cực.")
                    else:
                        st.warning(f"⚠️ Tiêu chí quan trọng nhất của bạn ({kw}) chỉ nhận được {score_val}/5 sao, đây là điểm cần cân nhắc kỹ trước khi quyết định gia nhập.")
                    break

            if eval_comp_info.get("avg_rating"):
                st.write(f"• **Điểm trung bình của bài review này:** `{pred_res['avg_score']} / 5.0` (So với điểm trung bình lịch sử của {eval_comp_info['Company Name']}: `{eval_comp_info['avg_rating']:.2f} / 5.0`).")

        # Radar / Bar breakdown comparing review to dataset benchmark
        st.write("")
        st.markdown("#### 📊 Đối chiếu 6 tiêu chí với trung vị ngành IT")
        chart_df = pd.DataFrame(
            {
                "Tiêu chí": [
                    "Rating",
                    "Lương & Phúc lợi",
                    "Đào tạo",
                    "Quản lý quan tâm",
                    "Văn hóa",
                    "Không gian",
                ],
                "Điểm bài review này": [s_rating, s_salary, s_train, s_mgmt, s_cult, s_off],
                "Trung vị toàn dataset ITViec": [
                    pred_res["median_benchmark"].get("Rating", 5.0),
                    pred_res["median_benchmark"].get("Salary & benefits", 4.0),
                    pred_res["median_benchmark"].get("Training & learning", 4.0),
                    pred_res["median_benchmark"].get("Management cares about me", 4.0),
                    pred_res["median_benchmark"].get("Culture & fun", 4.0),
                    pred_res["median_benchmark"].get("Office & workspace", 4.0),
                ],
            }
        ).set_index("Tiêu chí")

        st.bar_chart(chart_df, color=["#e11d48", "#94a3b8"])


# ==========================================
# PAGE 4: CHI TIẾT HỒ SƠ & REVIEW CÔNG TY (MỚI - BONUS)
# ==========================================
elif page == "📑 Hồ sơ & Review công ty":
    st.title("📑 Chi tiết hồ sơ & Toàn bộ Review công ty")
    st.write(
        "Khám phá hồ sơ chuyên sâu của từng doanh nghiệp công nghệ, phân tích biểu đồ 6 tiêu chí và đọc toàn bộ các bài đánh giá thực tế từ cựu và hiện nhân viên."
    )

    # Company selection
    if "target_company_view" in st.session_state and st.session_state["target_company_view"] in engine.id_to_index:
        st.session_state["prof_select_company"] = st.session_state.pop("target_company_view")
    if "prof_select_company" not in st.session_state or st.session_state["prof_select_company"] not in engine.id_to_index:
        st.session_state["prof_select_company"] = engine.company_ids[0]

    col_pick, _ = st.columns([3, 1])
    with col_pick:
        prof_company_id = st.selectbox(
            "Chọn công ty muốn xem hồ sơ chi tiết & review",
            options=engine.company_ids,
            format_func=engine.get_company_display_name,
            key="prof_select_company",
        )
    prof_comp = engine.get_company_by_id(prof_company_id)

    if prof_comp:
        # Full Company Profile Card
        with st.container(border=True):
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px;">
                    <div>
                        <div style="font-family:'Outfit', sans-serif; font-size:2rem; font-weight:800; color:#0f172a; line-height:1.2;">
                            {prof_comp['Company Name']}
                        </div>
                        <div style="margin-top:6px;">
                            <span class="chip chip-industry">🏢 {prof_comp.get('Company industry', 'IT')}</span>
                            <span class="chip chip-type">💼 {prof_comp.get('Company Type', 'IT')}</span>
                            <span class="chip chip-size">👥 {prof_comp.get('Company size', 'N/A')}</span>
                            <span class="chip chip-country">🌏 {prof_comp.get('Country', 'Vietnam')}</span>
                        </div>
                    </div>
                    <div>
                        {f'<div class="top-comp-badge" style="font-size:0.95rem; padding:6px 14px;">⭐ Rating: <b>' + str(prof_comp.get("avg_rating", "N/A")) + ' / 5.0</b></div>' if prof_comp.get("avg_rating") else '<span style="color:#64748b; font-size:0.85rem;">Chưa có review</span>'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            loc_val = prof_comp.get("Location")
            if pd.notna(loc_val) and str(loc_val).strip():
                st.markdown(f"📍 **Địa điểm văn phòng:** {loc_val}")

            work_val = prof_comp.get("Working days")
            ot_val = prof_comp.get("Overtime Policy")
            if (pd.notna(work_val) and str(work_val).strip()) or (pd.notna(ot_val) and str(ot_val).strip()):
                st.markdown(f"⏰ **Thời gian làm việc:** {work_val or 'N/A'} · **Chính sách OT:** {ot_val or 'N/A'}")

            skills_val = prof_comp.get("Our key skills")
            if pd.notna(skills_val) and str(skills_val).strip():
                st.markdown(f"💻 **Kỹ năng công nghệ chủ đạo (Key Skills):** `{skills_val}`")

            why_val = prof_comp.get("Why you'll love working here")
            if pd.notna(why_val) and str(why_val).strip():
                st.markdown(f"✨ **Lý do nhân viên yêu thích làm việc tại đây:**\n\n{why_val}")

            overview_val = prof_comp.get("Company overview")
            if pd.notna(overview_val) and str(overview_val).strip():
                with st.expander("📄 Xem toàn bộ mô hình kinh doanh & Tổng quan (Overview)", expanded=False):
                    st.write(overview_val)

            href_val = prof_comp.get("Href")
            if pd.notna(href_val) and str(href_val).strip():
                st.markdown(f"[🔗 Trang chủ tuyển dụng trên ITViec]({href_val})")

        # Ratings breakdown if reviews exist
        reviews_df = engine.get_company_reviews(prof_company_id)

        st.write("")
        if reviews_df.empty:
            st.info(f"Hiện chưa có bài review nào cho công ty {prof_comp['Company Name']} trong tập dữ liệu 8.417 review.")
        else:
            st.subheader(f"📊 Thống kê đánh giá của nhân viên ({len(reviews_df)} bài review)")

            m_r1, m_r2, m_r3, m_r4, m_r5, m_r6 = st.columns(6)
            m_r1.metric("Rating chung", f"{prof_comp['avg_rating']:.2f} ⭐")
            m_r2.metric("Lương & Thưởng", f"{prof_comp['avg_salary']:.2f}")
            m_r3.metric("Đào tạo", f"{prof_comp['avg_training']:.2f}")
            m_r4.metric("Ban quản lý", f"{prof_comp['avg_management']:.2f}")
            m_r5.metric("Văn hóa", f"{prof_comp['avg_culture']:.2f}")
            m_r6.metric("Không gian", f"{prof_comp['avg_office']:.2f}")

            # Visual comparison bar chart against IT industry medians
            st.write("")
            st.markdown("##### 📈 So sánh chỉ số trung bình của công ty với toàn ngành IT")
            comp_metrics_df = pd.DataFrame(
                {
                    "Tiêu chí": ["Rating", "Lương & Phúc lợi", "Đào tạo", "Ban quản lý", "Văn hóa", "Không gian"],
                    f"{prof_comp['Company Name']}": [
                        prof_comp["avg_rating"],
                        prof_comp["avg_salary"],
                        prof_comp["avg_training"],
                        prof_comp["avg_management"],
                        prof_comp["avg_culture"],
                        prof_comp["avg_office"],
                    ],
                    "Trung vị ngành IT": [5.0, 4.0, 4.0, 4.0, 4.0, 4.0],
                }
            ).set_index("Tiêu chí")
            st.bar_chart(comp_metrics_df, color=["#e11d48", "#94a3b8"])

            # REVIEWS LIST SECTION
            st.write("")
            st.subheader(f"💬 Danh sách toàn bộ các bài review ({len(reviews_df)} đánh giá)")

            col_fil_rec, col_fil_star = st.columns([2, 2])
            with col_fil_rec:
                filter_rec_label = st.radio(
                    "Lọc theo nhãn Recommend:",
                    ["Tất cả", "Chỉ xem Recommend (Yes)", "Chỉ xem Not Recommend (No)"],
                    horizontal=True,
                    key="filter_rev_recommend",
                )
            with col_fil_star:
                filter_star = st.selectbox(
                    "Lọc theo số sao Rating:",
                    ["Tất cả số sao", "5 sao", "4 sao", "3 sao", "2 sao", "1 sao"],
                    key="filter_rev_star",
                )

            # Apply filters
            filtered_revs = reviews_df.copy()
            if filter_rec_label == "Chỉ xem Recommend (Yes)":
                filtered_revs = filtered_revs[filtered_revs["Recommend?"].astype(str).str.strip().str.capitalize() == "Yes"]
            elif filter_rec_label == "Chỉ xem Not Recommend (No)":
                filtered_revs = filtered_revs[filtered_revs["Recommend?"].astype(str).str.strip().str.capitalize() == "No"]

            if filter_star != "Tất cả số sao":
                target_s = int(filter_star.split()[0])
                filtered_revs = filtered_revs[filtered_revs["Rating"] == target_s]

            st.caption(f"Đang hiển thị **{len(filtered_revs)} / {len(reviews_df)}** bài review thỏa mãn bộ lọc:")

            for rev_idx, rev_row in filtered_revs.iterrows():
                is_yes = str(rev_row.get("Recommend?", "")).strip().capitalize() == "Yes"
                st.markdown(
                    f"""
                    <div class="review-card">
                        <div class="review-top-bar">
                            <div>
                                <span class="review-title">{rev_row.get('Title', 'Không có tiêu đề')}</span>
                                <span class="star-rating" style="margin-left:10px;">⭐ {rev_row.get('Rating', 5)}/5</span>
                            </div>
                            <div>
                                <span class="{'review-badge-yes' if is_yes else 'review-badge-no'}">
                                    {'👍 Recommend (Khuyên chọn)' if is_yes else '👎 Not Recommend (Không khuyên)'}
                                </span>
                            </div>
                        </div>
                        <div class="review-date">
                            📅 Thời gian đăng: {rev_row.get('Cmt_day', 'N/A')} · Lương: {rev_row.get('Salary & benefits', 4)}/5 · Đào tạo: {rev_row.get('Training & learning', 4)}/5 · Quản lý: {rev_row.get('Management cares about me', 4)}/5 · Văn hóa: {rev_row.get('Culture & fun', 4)}/5 · Văn phòng: {rev_row.get('Office & workspace', 4)}/5
                        </div>
                        <div class="review-quote-box">
                            <b>👍 Điều thích nhất:</b> {rev_row.get('What I liked', 'Không có ý kiến')}
                        </div>
                        <div class="review-quote-box" style="border-left-color:#e11d48;">
                            <b>💡 Đề xuất cải thiện:</b> {rev_row.get('Suggestions for improvement', 'Không có ý kiến')}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Export reviews of this company
            st.download_button(
                f"↓ Tải toàn bộ review của {prof_comp['Company Name']} (CSV)",
                download_csv_bytes(reviews_df),
                file_name=f"reviews_{prof_comp['Company Name'].replace(' ', '_').lower()}.csv",
                mime="text/csv",
                key="btn_dl_company_reviews",
            )

        # BRAINSTORM FEATURE: HEAD-TO-HEAD COMPARISON
        st.write("")
        with st.expander("🥊 Tính năng mở rộng: So sánh đối đầu giữa 2 công ty (Head-to-Head Comparison)", expanded=False):
            h2h_candidates = [cid for cid in engine.company_ids if cid != prof_company_id]
            c2_id = st.selectbox(
                "Chọn công ty đối thủ / tham chiếu để so sánh",
                options=h2h_candidates,
                format_func=engine.get_company_display_name,
                key="h2h_select_comp2",
            )
            c2_info = engine.get_company_by_id(c2_id)

            if c2_info:
                # Compute direct pairwise similarity between c1 and c2
                idx1 = engine.id_to_index.get(prof_company_id)
                idx2 = engine.id_to_index.get(c2_id)
                direct_sim = 0.0
                if idx1 is not None and idx2 is not None:
                    direct_sim = float((engine.rec_matrix[idx1] @ engine.rec_matrix[idx2].T).toarray()[0, 0]) * 100

                st.markdown(
                    f"""
                    <div style="background:#f8fafc; border:1px solid #cbd5e1; border-radius:12px; padding:16px; margin:12px 0;">
                        <div style="text-align:center; font-family:'Outfit',sans-serif; font-size:1.1rem; font-weight:750; color:#475569;">
                            ĐỘ TƯƠNG ĐỒNG NỘI DUNG GIỮA 2 CÔNG TY: <span style="color:#e11d48; font-size:1.3rem;">{direct_sim:.1f}%</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                h_col1, h_col2 = st.columns(2)
                with h_col1:
                    with st.container(border=True):
                        st.markdown(f"### {prof_comp['Company Name']}")
                        st.markdown(f"• **Loại hình:** {prof_comp.get('Company Type', 'IT')}")
                        st.markdown(f"• **Ngành nghề:** {prof_comp.get('Company industry', 'IT')}")
                        st.markdown(f"• **Quy mô:** {prof_comp.get('Company size', 'N/A')}")
                        st.markdown(f"• **Quốc gia:** {prof_comp.get('Country', 'N/A')}")
                        st.markdown(f"• ⭐ **Rating:** `{prof_comp.get('avg_rating', 'Chưa có review')}`")
                        st.markdown(f"• 👍 **Recommend:** `{prof_comp.get('pct_recommend', 'N/A')}%`")
                        if prof_comp.get("Our key skills"):
                            st.markdown(f"• **Key skills:** `{prof_comp['Our key skills']}`")
                with h_col2:
                    with st.container(border=True):
                        st.markdown(f"### {c2_info['Company Name']}")
                        st.markdown(f"• **Loại hình:** {c2_info.get('Company Type', 'IT')}")
                        st.markdown(f"• **Ngành nghề:** {c2_info.get('Company industry', 'IT')}")
                        st.markdown(f"• **Quy mô:** {c2_info.get('Company size', 'N/A')}")
                        st.markdown(f"• **Quốc gia:** {c2_info.get('Country', 'N/A')}")
                        st.markdown(f"• ⭐ **Rating:** `{c2_info.get('avg_rating', 'Chưa có review')}`")
                        st.markdown(f"• 👍 **Recommend:** `{c2_info.get('pct_recommend', 'N/A')}%`")
                        if c2_info.get("Our key skills"):
                            st.markdown(f"• **Key skills:** `{c2_info['Our key skills']}`")


# ==========================================
# PAGE 5: DỰ ĐOÁN HÀNG LOẠT (BATCH REVIEW CSV)
# ==========================================
elif page == "📁 Dự đoán hàng loạt":
    st.title("📁 Dự đoán hàng loạt từ file (CSV / Excel)")
    st.write(
        "Tải lên tệp chứa nhiều bài review để hệ thống tự động trích xuất đặc trưng và dự đoán nhãn Recommend hàng loạt."
    )

    up_col, sample_col = st.columns([3, 2])

    with up_col:
        uploaded_file = st.file_uploader(
            "Chọn file review (CSV hoặc Excel)",
            type=["csv", "xlsx", "xls"],
            key="batch_file_uploader",
        )

    with sample_col:
        st.write("**Tải file mẫu thử nghiệm:**")
        sample_no_label_path = ROOT / "samples" / "mau_review_khong_nhan.csv"
        sample_with_label_path = ROOT / "samples" / "mau_review_co_nhan.csv"

        if sample_no_label_path.exists():
            st.download_button(
                "↓ Tải mẫu CSV (Chưa có nhãn)",
                sample_no_label_path.read_bytes(),
                file_name="mau_review_khong_nhan.csv",
                mime="text/csv",
                key="dl_sample_no_label",
            )
        if sample_with_label_path.exists():
            st.download_button(
                "↓ Tải mẫu CSV (Kèm nhãn thực tế để đối chiếu)",
                sample_with_label_path.read_bytes(),
                file_name="mau_review_co_nhan.csv",
                mime="text/csv",
                key="dl_sample_with_label",
            )

        use_demo_data = st.checkbox("Dùng dữ liệu mẫu tích hợp sẵn để thử ngay", key="chk_use_demo_batch")

    df_to_process: pd.DataFrame | None = None

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_to_process = pd.read_csv(uploaded_file)
            else:
                df_to_process = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Không thể đọc file: {e}")
    elif use_demo_data and sample_with_label_path.exists():
        df_to_process = pd.read_csv(sample_with_label_path)
        st.info("💡 Đang sử dụng 25 bài review mẫu trích xuất ngẫu nhiên từ tập dữ liệu học tập.")

    if df_to_process is not None:
        st.write("")
        st.caption(f"Tệp chứa **{len(df_to_process):,}** bài review · **{len(df_to_process.columns)}** cột. Xem trước 5 dòng đầu:")
        st.dataframe(df_to_process.head(5), hide_index=True, width="stretch")

        if st.button("Tiến hành dự đoán toàn bộ file 🚀", type="primary", key="btn_run_batch"):
            with st.spinner("Đang tiền xử lý văn bản và chạy mô hình học máy..."):
                result_df, eval_metrics = engine.predict_batch_reviews(df_to_process)

            st.success(f"✓ Đã dự đoán thành công cho toàn bộ {len(result_df):,} review!")

            # Summary metrics
            n_yes = int((result_df["du_doan_recommend"] == "Yes").sum())
            n_no = int((result_df["du_doan_recommend"] == "No").sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Tổng số review", f"{len(result_df):,}")
            m2.metric("Dự đoán Recommend (Yes)", f"{n_yes:,} ({n_yes / len(result_df):.1%})")
            m3.metric("Dự đoán Not Recommend (No)", f"{n_no:,} ({n_no / len(result_df):.1%})")

            # Evaluation if actual labels existed
            if eval_metrics:
                st.write("")
                with st.container(border=True):
                    st.markdown("#### 🎯 Đối chiếu với nhãn thực tế trong tệp (Ground Truth)")
                    em1, em2, em3, em4 = st.columns(4)
                    em1.metric("Độ chính xác (Accuracy)", f"{eval_metrics['accuracy']:.1%}")
                    em2.metric("Precision (Lớp No)", f"{eval_metrics['precision_no']:.3f}")
                    em3.metric("Recall (Lớp No)", f"{eval_metrics['recall_no']:.3f}")
                    em4.metric("F1-score (Lớp No)", f"{eval_metrics['f1_no']:.3f}")

                    # Confusion Matrix mini table
                    cm = eval_metrics["confusion_matrix"]
                    st.write("**Ma trận nhầm lẫn (Confusion Matrix):**")
                    cm_df = pd.DataFrame(
                        {
                            "Dự đoán Yes": [cm[0][0], cm[1][0]],
                            "Dự đoán No": [cm[0][1], cm[1][1]],
                        },
                        index=["Thực tế Yes", "Thực tế No"],
                    )
                    st.dataframe(cm_df, width="stretch")

            # Filter options & Table display
            st.write("")
            filter_only_no = st.checkbox("Chỉ xem các review bị dự đoán 'No' (Cần kiểm tra)", key="chk_filter_no")
            display_df = result_df[result_df["du_doan_recommend"] == "No"] if filter_only_no else result_df

            st.dataframe(display_df, hide_index=True, width="stretch")

            # Safe download
            csv_bytes = download_csv_bytes(result_df)
            st.download_button(
                "↓ Tải toàn bộ kết quả đã dự đoán (CSV UTF-8)",
                csv_bytes,
                file_name="ket_qua_du_doan_itviec_reviews.csv",
                mime="text/csv",
                type="primary",
                key="btn_dl_batch_results",
            )


# ==========================================
# PAGE 6: BÁO CÁO MÔ HÌNH & BENCHMARK
# ==========================================
elif page == "🔬 Báo cáo mô hình":
    st.title("🔬 Báo cáo kỹ thuật & Đánh giá mô hình học máy")
    st.write(
        "Tổng hợp đầy đủ quy trình nghiên cứu Data Science Process, kết quả so sánh giữa các thuật toán và xử lý mất cân bằng nhãn."
    )

    tab_eval, tab_plots, tab_limits = st.tabs(["📊 Bảng so sánh mô hình", "📈 Bộ sưu tập biểu đồ", "⚠️ Giới hạn & Khuyến nghị"])

    # --- TAB 1: BẢNG SO SÁNH MÔ HÌNH ---
    with tab_eval:
        st.subheader("1. So sánh các mô hình phân loại trên tập kiểm thử (Test Set)")
        st.write(
            "Tập test được chia theo **công ty** (GroupShuffleSplit theo `id` công ty) với 36 công ty và 3.212 review hoàn toàn mới lạ. "
            "Chỉ số mục tiêu tối thượng là **F1-score cho lớp 'No'** do nhãn mất cân bằng nặng (12.3% No)."
        )

        st.markdown("#### A. Nhóm mô hình Scikit-Learn (Python)")
        sk_test_data = [
            {"Model": "DummyMostFrequent", "Accuracy": 0.838, "Precision No": 0.000, "Recall No": 0.000, "F1-score No": 0.000, "ROC-AUC No": 0.500, "Ghi chú": "Mô hình cơ sở ngẫu nhiên"},
            {"Model": "LogisticRegression", "Accuracy": 0.811, "Precision No": 0.457, "Recall No": 0.893, "F1-score No": 0.605, "ROC-AUC No": 0.932, "Ghi chú": "Class weight balanced"},
            {"Model": "LinearSVC", "Accuracy": 0.883, "Precision No": 0.618, "Recall No": 0.722, "F1-score No": 0.666, "ROC-AUC No": 0.925, "Ghi chú": "Hiệu năng phân loại rất cao"},
            {"Model": "KNN (Cosine)", "Accuracy": 0.839, "Precision No": 1.000, "Recall No": 0.006, "F1-score No": 0.011, "ROC-AUC No": 0.548, "Ghi chú": "Bị áp đảo bởi khoảng cách"},
            {"Model": "RandomForest (Selected)", "Accuracy": 0.867, "Precision No": 0.570, "Recall No": 0.733, "F1-score No": 0.641, "ROC-AUC No": 0.913, "Ghi chú": "Mô hình chọn trên Validation (150 cây)"},
        ]
        df_sk_table = pd.DataFrame(sk_test_data)
        st.dataframe(df_sk_table, hide_index=True, width="stretch")

        st.write("")
        st.markdown("#### B. Nhóm mô hình Phân tán PySpark ML")
        spark_test_data = [
            {"Model": "Spark LogisticRegression", "Accuracy": 0.852, "Precision No": 0.535, "Recall No": 0.655, "F1-score No": 0.589, "ROC-AUC No": 0.865, "Ghi chú": "Có trọng số lớp"},
            {"Model": "Spark DecisionTree", "Accuracy": 0.907, "Precision No": 0.872, "Recall No": 0.499, "F1-score No": 0.635, "ROC-AUC No": 0.884, "Ghi chú": "Độ chính xác thô cao"},
            {"Model": "Spark RandomForest", "Accuracy": 0.781, "Precision No": 0.417, "Recall No": 0.887, "F1-score No": 0.568, "ROC-AUC No": 0.887, "Ghi chú": "Có trọng số lớp"},
            {"Model": "Spark GBT (Selected Best)", "Accuracy": 0.910, "Precision No": 0.825, "Recall No": 0.562, "F1-score No": 0.669, "ROC-AUC No": 0.922, "Ghi chú": "Mô hình tốt nhất trên PySpark"},
        ]
        df_spark_table = pd.DataFrame(spark_test_data)
        st.dataframe(df_spark_table, hide_index=True, width="stretch")

        st.write("")
        st.markdown("#### C. Đối chiếu hệ thống gợi ý: Scikit-Learn vs Gensim")
        rec_m = engine.recommendation_metrics
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Đặc trưng Sklearn TF-IDF", f"{rec_m['sklearn_features']:,}")
        rc2.metric("Đặc trưng Gensim", f"{rec_m['gensim_features']:,}")
        rc3.metric("Độ trùng Top-5 trung bình", f"{rec_m['mean_top5_overlap'] * 100:.1f}%")

    # --- TAB 2: BỘ SƯU TẬP BIỂU ĐỒ ---
    with tab_plots:
        st.subheader("2. Biểu đồ đánh giá học máy trực quan")
        img_dir = ROOT / "assets" / "images"

        plot_type = st.radio(
            "Chọn nhóm biểu đồ muốn xem:",
            [
                "Ma trận nhầm lẫn & Đường cong ROC",
                "Đặc trưng quan trọng (Feature Importance)",
                "Word Cloud từ khóa đánh giá",
                "Phân tích lệch nhãn (Imbalance) & So sánh Gensim",
            ],
            horizontal=True,
            key="radio_plot_group",
        )

        if plot_type == "Ma trận nhầm lẫn & Đường cong ROC":
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.write("**Ma trận nhầm lẫn (Test Set - Random Forest)**")
                cm_img = img_dir / "confusion_matrix.png"
                if cm_img.exists():
                    st.image(str(cm_img), width="stretch")
            with p_col2:
                st.write("**Đường cong ROC (Lớp No)**")
                roc_img = img_dir / "roc_curve.png"
                if roc_img.exists():
                    st.image(str(roc_img), width="stretch")

            # Combined LinearSVC and GBT plots if available
            cm_gbt = img_dir / "confusion_matrix_linearsvc_gbt.png"
            if cm_gbt.exists():
                st.write("**Đối chiếu LinearSVC vs PySpark GBT:**")
                st.image(str(cm_gbt), width="stretch")

        elif plot_type == "Đặc trưng quan trọng (Feature Importance)":
            fi_col1, fi_col2 = st.columns(2)
            with fi_col1:
                st.write("**Tầm quan trọng đặc trưng (Scikit-Learn Random Forest)**")
                fi_sk = img_dir / "feature_importance_sklearn.png"
                if fi_sk.exists():
                    st.image(str(fi_sk), width="stretch")
            with fi_col2:
                st.write("**Trọng số hệ số (LinearSVC)**")
                fi_svc = img_dir / "feature_importance_linearsvc.png"
                if fi_svc.exists():
                    st.image(str(fi_svc), width="stretch")

        elif plot_type == "Word Cloud từ khóa đánh giá":
            st.write("**Đám mây từ khóa nổi bật theo nhãn Recommend (Yes vs No)**")
            wc_img = img_dir / "wordcloud_recommend.png"
            if wc_img.exists():
                st.image(str(wc_img), width="stretch")
            else:
                st.info("Biểu đồ Word Cloud đang được đồng bộ.")

        elif plot_type == "Phân tích lệch nhãn (Imbalance) & So sánh Gensim":
            imb_col1, imb_col2 = st.columns(2)
            with imb_col1:
                st.write("**So sánh chiến lược xử lý mất cân bằng**")
                imb_img = img_dir / "imbalance_comparison.png"
                if imb_img.exists():
                    st.image(str(imb_img), width="stretch")
            with imb_col2:
                st.write("**So sánh độ tương đồng: Sklearn vs Gensim**")
                gs_img = img_dir / "compare_sklearn_gensim.png"
                if gs_img.exists():
                    st.image(str(gs_img), width="stretch")

    # --- TAB 3: GIỚI HẠN & KHUYẾN NGHỊ ---
    with tab_limits:
        st.subheader("3. Giới hạn mô hình và Đề xuất triển khai")
        st.markdown(
            """
            > [!NOTE]
            > **1. Rò rỉ dữ liệu (Data Leakage) đã được kiểm soát chặt chẽ:**
            > Việc chia train/validation/test được phân cụm theo `id công ty` (GroupShuffleSplit). Điều này đảm bảo toàn bộ review của cùng một công ty chỉ nằm trong duy nhất 1 tập, phản ánh chính xác khả năng tổng quát hóa của mô hình khi gặp một công ty hoàn toàn mới.

            > [!IMPORTANT]
            > **2. Mất cân bằng nhãn dữ liệu (Class Imbalance):**
            > Tỷ lệ review 'No' chỉ chiếm 12.3%. Nếu chỉ nhìn vào Accuracy thông thường (83-91%), mô hình Dummy đoán toàn bộ 'Yes' cũng đã đạt 83.8% nhưng hoàn toàn vô dụng. Do đó, việc áp dụng trọng số lớp (`class_weight='balanced'`) và đo lường F1-score trên lớp No là quyết định kỹ thuật chuẩn mực.

            > [!TIP]
            > **3. Đề xuất mở rộng trong tương lai:**
            > - **Mô hình ngôn ngữ tiền huấn luyện (Pre-trained Transformers):** Thử nghiệm mô hình PhởBERT hoặc ViDeBERTa để nắm bắt ngữ nghĩa và sự mỉa mai (sarcasm) trong các bài review tiếng Việt tốt hơn TF-IDF.
            > - **Tương tác đa chiều (Collaborative Filtering):** Khi có thêm dữ liệu hành vi ứng viên xem tin tuyển dụng, có thể phát triển mô hình Hybrid Recommendation kết hợp giữa Content-Based và Collaborative Filtering.
            """
        )
