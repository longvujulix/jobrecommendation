# GUI Project 2 · ITViec Hub: Career Match & Review Sentiment AI

Ứng dụng web Streamlit độc lập cho **Topic 2: Đề xuất công ty dựa trên nội dung & Phân loại Review ('Recommend or Not')**, dựa trên yêu cầu trong `GUI/` và dữ liệu đồ án ITViec (`Project 2/`). Ứng dụng chạy độc lập, gọn nhẹ, đã tích hợp sẵn toàn bộ mô hình và tài nguyên, **không bắt buộc cài đặt Spark/Java** khi chạy demo giao diện.

---

## 🚀 Khởi động ứng dụng trên máy tính

Mở terminal hoặc dòng lệnh tại thư mục `GUI_Project2`:

### 1. Trên Windows (1 lệnh duy nhất hoặc click đúp):
```cmd
setup.bat
```
*Hoặc khởi động trực tiếp qua Streamlit:*
```cmd
python -m streamlit run app.py
```

### 2. Trên macOS / Linux:
```bash
bash setup.sh
```
*Hoặc khởi động trực tiếp:*
```bash
python3 -m streamlit run app.py
```

Ứng dụng sẽ tự động mở giao diện tại địa chỉ: `http://localhost:8501`. File `setup.sh` và `setup.bat` đã tích hợp sẵn cơ chế kiểm tra môi trường: nếu máy đã có đủ thư viện sẽ chạy ngay lập tức, nếu thiếu sẽ tự động cài từ `requirements.txt`.

---

## 🖥️ Các không gian làm việc trên giao diện

Giao diện được thiết kế hiện đại theo phong cách nhận diện công nghệ ITViec (**Crimson Rose `#e11d48` kết hợp Tech Indigo `#4f46e5`**), gồm 5 khu vực làm việc:

1. **🌐 Tổng quan (Overview & Ecosystem Dashboard)**:
   - Thống kê toàn cảnh dữ liệu đồ án: 478 công ty CNTT, 8.417 review thực tế, 9.891 chiều đặc trưng TF-IDF.
   - Thẻ điều hướng nhanh đến các khu vực làm việc chính.
   - Biểu đồ phân tích dữ liệu khám phá (EDA): Phân bố nhãn `Recommend?` (bài toán mất cân bằng 87.7% Yes vs 12.3% No) và phân bố Rating 1–5 sao theo nhãn.

2. **🏢 Gợi ý công ty (Content-Based Recommender System)**:
   - **Chế độ 1 - Theo công ty mục tiêu:** Chọn một trong 478 công ty, hệ thống hiển thị hồ sơ chi tiết (Ngành nghề, loại hình Product/Outsourcing, quy mô, quốc gia, kỹ năng nổi bật, lý do thích làm việc) và đề xuất **Top-K công ty tương đồng nhất** kèm điểm % Cosine Similarity. Kèm bảng đối chiếu thực nghiệm giữa **Scikit-Learn TF-IDF** và **Gensim Topic Model**.
   - **Chế độ 2 - Tìm theo kỹ năng ứng viên:** Ứng viên nhập văn bản tự do chứa kỹ năng công nghệ hoặc kỳ vọng môi trường (ví dụ: `Python, React, microservices, cloud AWS, hybrid working`), hệ thống tính vector TF-IDF và quét cosine tìm các công ty khớp nhất.

3. **✍️ Đánh giá review (Single Review Classifier)**:
   - Form nhập văn bản: Tiêu đề (`Title`), Điểm thích nhất (`What I liked`), Đề xuất cải thiện (`Suggestions for improvement`).
   - 6 Thang đo điểm số (1–5 sao): `Rating`, `Salary & benefits`, `Training & learning`, `Management cares about me`, `Culture & fun`, `Office & workspace`.
   - Cung cấp nút thử nhanh với mẫu review tích cực và mẫu review tiêu cực.
   - Dự đoán tức thì nhãn **RECOMMEND (Yes)** hoặc **NOT RECOMMEND (No)** kèm phần trăm độ tin cậy (Confidence %) và biểu đồ đối chiếu 6 tiêu chí với trung vị toàn ngành IT.

4. **📁 Dự đoán hàng loạt (Batch Processing CSV / Excel)**:
   - Tải lên tệp danh sách review định dạng `.csv` hoặc `.xlsx`.
   - Cung cấp sẵn 2 nút tải file mẫu (`mau_review_khong_nhan.csv` và `mau_review_co_nhan.csv`) cùng tùy chọn dùng dữ liệu mẫu minh họa.
   - Tự động chuẩn hóa và nhận diện linh hoạt tên cột tiếng Anh và tiếng Việt.
   - Dự đoán toàn bộ tệp, hiển thị số lượng và biểu đồ phân bổ nhãn.
   - Nếu tệp có sẵn nhãn thực tế (`Recommend?`): Tự động tính toán Accuracy, Precision (No), Recall (No), F1-score (No) và bảng Ma trận nhầm lẫn (Confusion Matrix).
   - Bộ lọc xem riêng các review bị gắn cờ "No".
   - Tải kết quả CSV an toàn định dạng UTF-8-BOM (chống lỗi tiếng Việt trên Excel và chống formula injection).

5. **🔬 Báo cáo mô hình (Technical Report & Model Benchmarks)**:
   - Quy trình Data Science Lifecycle: Tiền xử lý NLP tiếng Việt, tách nhóm theo `id` công ty (GroupShuffleSplit) để tránh rò rỉ dữ liệu (data leakage).
   - Bảng so sánh 5 mô hình Scikit-Learn (`Dummy`, `LogisticRegression`, `LinearSVC`, `KNN`, `RandomForest`) và 4 mô hình PySpark ML (`LogisticRegression`, `DecisionTree`, `RandomForest`, `GBT`).
   - Phân tích giải pháp xử lý mất cân bằng nhãn (`class_weight='balanced'`).
   - Bộ sưu tập biểu đồ học máy: Confusion Matrix, ROC Curve, Feature Importance, Word Cloud, So sánh Sklearn vs Gensim.
   - Hạn chế và khuyến nghị mở rộng (Transformers, Hybrid Recommender).

---

## 📦 Kiến trúc tài nguyên và mô hình đã đóng gói

Toàn bộ tài nguyên được lưu trữ độc lập trong `assets/` và `samples/`, không phụ thuộc vào mã nguồn bên ngoài:

- `assets/models/recommender.joblib` (1.67 MB): Cấu trúc ma trận TF-IDF 478 công ty (9.891 đặc trưng n-gram) và vectorizer đã fit.
- `assets/models/classifier.joblib` (4.40 MB): Mô hình Random Forest 150 cây kèm bộ trích xuất TF-IDF và thông số điền thiếu median chuẩn hóa.
- `assets/companies.parquet` (424 KB): Dữ liệu đầy đủ của 478 công ty (ngành, quy mô, kỹ năng, lý do làm việc, địa chỉ, link ITViec).
- `assets/classification.json`, `recommendation.json`, `eda.json`: Toàn bộ chỉ số thực nghiệm trên tập Test và đối chiếu Gensim.
- `assets/images/*.png`: 10+ biểu đồ học máy được trích xuất từ quá trình huấn luyện pipeline.

Khi cần cập nhật hoặc đóng gói lại từ thư mục `Project 2`:
```bash
python export_assets.py
# Hoặc chỉ định đường dẫn: python export_assets.py --source /duong/dan/toi/Project_2
```

---

## 📋 Yêu cầu cấu trúc tệp khi tải lên tệp hàng loạt (Batch CSV / Excel)

Ứng dụng hỗ trợ linh hoạt cả tên cột tiếng Anh chuẩn của ITViec lẫn tiếng Việt:

| Thông tin | Tên cột tiếng Anh được hỗ trợ | Tên cột tiếng Việt được hỗ trợ |
| :--- | :--- | :--- |
| **Tiêu đề review** | `Title`, `title` | `tieu_de`, `Tiêu đề` |
| **Điểm thích** | `What I liked`, `liked` | `thich`, `Điểm thích` |
| **Đề xuất cải thiện** | `Suggestions for improvement`, `suggestions` | `gop_y`, `Đề xuất`, `Cải thiện` |
| **Điểm tổng quan** | `Rating`, `rating` | `diem_tong_quan`, `Đánh giá` |
| **Lương & Phúc lợi** | `Salary & benefits` | `luong_thuong`, `phuc_loi` |
| **Đào tạo & Học hỏi** | `Training & learning` | `dao_tao`, `hoc_hoi` |
| **Sếp quan tâm** | `Management cares about me` | `quan_ly_quan_tam`, `sep_quan_tam` |
| **Văn hóa & Vui vẻ** | `Culture & fun` | `van_hoa`, `niem_vui` |
| **Văn phòng & Thiết bị**| `Office & workspace` | `van_phong`, `co_so_vat_chat` |
| **Nhãn thực tế (tùy chọn)**| `Recommend?`, `recommend`, `label` | `nhan_xet`, `khuyen_chon` |

*Lưu ý: Nếu một số tiêu chí điểm bị thiếu, mô hình sẽ tự động sử dụng trung vị ngành (`numeric_median`) đã học từ tập huấn luyện.*

---

## 🌐 Xuất bản lên Streamlit Community Cloud

Thư mục `GUI_Project2` có cấu trúc chuẩn mực để deploy trực tiếp lên [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Tạo một repository trên GitHub (hoặc đưa vào nhánh của repo hiện tại).
2. Thiết lập đường dẫn Root/Working Directory trỏ vào thư mục `GUI_Project2`.
3. Chọn file chạy chính: `app.py`.
4. Môi trường: Python 3.10 – 3.12 (Streamlit Cloud sẽ tự động cài các thư viện trong `requirements.txt`).
5. Không cần cài đặt Java hay Spark; mô hình scikit-learn sẽ chạy tức thì với tốc độ cao.
