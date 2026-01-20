@echo off
cd /d "%~dp0"
echo Starting QuantFolio Dashboard...
streamlit run app.py
pause
