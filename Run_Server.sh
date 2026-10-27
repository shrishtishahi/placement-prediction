#!/usr/bin/env bash
cd "$(dirname "$0")"
echo
echo " Student Placement Prediction"
echo " ----------------------------"
echo

PY=""
for candidate in python3.12 python3.11 python3.13 python3.10 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PY="$candidate"
    break
  fi
done

if [ -z "$PY" ]; then
  echo "Python was not found."
  echo "Install Python 3.11 or newer, then run this file again."
  exit 1
fi

echo "Using $PY"
if [ ! -x ".venv/bin/python" ]; then
  echo "Creating a local environment. This happens only once."
  "$PY" -m venv .venv || exit 1
fi

echo "Checking required libraries..."
.venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt || exit 1

echo
echo "Opening the app in your browser."
echo "Keep this window open while you present."
echo
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_HEADLESS=true
(sleep 2 && (xdg-open http://localhost:8501 >/dev/null 2>&1 || open http://localhost:8501 >/dev/null 2>&1 || true)) &
exec .venv/bin/python -m streamlit run app.py --browser.gatherUsageStats=false --server.headless=true --server.port=8501
