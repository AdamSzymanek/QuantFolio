import subprocess
import os
import sys

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
app_path = os.path.join(current_dir, "app.py")

print(f"Launching Streamlit App: {app_path}")
print("Please wait...")

# Launch streamlit run app.py
subprocess.run(["streamlit", "run", app_path], check=True)
