# Student Placement Prediction

Streamlit app that predicts whether a student will be placed.

## Run on any computer

1. Install Python 3.11 or newer from https://www.python.org/downloads/
   Tick **Add python.exe to PATH**.
2. Extract this folder.
3. On Windows, double-click **Run_Server.bat**.
   On Mac or Linux, run `bash Run_Server.sh`.
4. Leave that window open. The app opens in the browser.
5. Close the window when you are finished.

The first launch needs internet to install the libraries. Later launches start straight away.

## Files

- `app.py` — Streamlit pages
- `predictor.py` — trained neural network
- `report_pdf.py` — PDF report
- `ann_weights.npz` — saved model weights
- `Placement_Prediction_data.csv` — dataset shown on the home page
- `requirements.txt` — libraries
- `Run_Server.bat` / `Run_Server.sh` — one-click start
