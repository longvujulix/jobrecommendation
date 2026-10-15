#!/usr/bin/env bash
# Script kiểm tra môi trường và cài đặt cho GUI Project 2 (macOS / Linux)
set -e

# 1. Kiểm tra Python
if command -v python3 >/dev/null 2>&1; then
    PY_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PY_CMD="python"
else
    echo "[LỖI] Không tìm thấy Python trên hệ thống. Vui lòng cài đặt Python 3.10+."
    exit 1
fi

echo "Đang sử dụng: $($PY_CMD --version)"

# 2. Kiểm tra các thư viện cần thiết
echo "Đang kiểm tra các thư viện bắt buộc..."
if $PY_CMD -c "import streamlit, pandas, scipy, sklearn, joblib, pyarrow, numpy" >/dev/null 2>&1; then
    echo "✓ Tất cả thư viện (streamlit, pandas, scipy, sklearn, joblib, pyarrow, numpy) đã có sẵn."
else
    echo "Thiếu thư viện, đang cài đặt từ requirements.txt..."
    $PY_CMD -m pip install -r requirements.txt
fi

# 3. Cấu hình cho môi trường Cloud có biến $PORT (nếu có)
if [ -n "${PORT:-}" ]; then
    mkdir -p ~/.streamlit/
    cat <<EOF > ~/.streamlit/config.toml
[server]
headless = true
port = $PORT
enableCORS = false
EOF
fi

echo "✓ Môi trường đã sẵn sàng!"
echo "Chạy ứng dụng bằng lệnh: $PY_CMD -m streamlit run app.py"
$PY_CMD -m streamlit run app.py
