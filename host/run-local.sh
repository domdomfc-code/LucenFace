#!/usr/bin/env bash
# Chạy Streamlit LucenFace trên localhost (macOS / Linux).
set -euo pipefail

HOST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HOST_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "Repo: ${REPO_ROOT}"

VENV_PY="${REPO_ROOT}/.venv/bin/python"
VENV_PIP="${REPO_ROOT}/.venv/bin/pip"

if [[ ! -x "${VENV_PY}" ]]; then
  echo "Tạo venv .venv ..."
  python3 -m venv .venv
fi

echo "Cài requirements.txt (có thể vài phút)..."
"${VENV_PIP}" install -U pip
"${VENV_PIP}" install -r "${REPO_ROOT}/requirements.txt"

echo ""
echo "Mở trình duyệt: http://localhost:8501"
echo "Dừng: Ctrl+C"
echo ""

exec "${REPO_ROOT}/.venv/bin/streamlit" run "${REPO_ROOT}/app.py" \
  --server.address=localhost \
  --server.port=8501 \
  --browser.gatherUsageStats=false
