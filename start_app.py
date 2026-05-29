import subprocess
import webbrowser
import time
import os
import sys

# get correct path (works in EXE + normal run)
if getattr(sys, 'frozen', False):
    base_dir = os.path.dirname(sys.executable)
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))

app_path = os.path.join(base_dir, "app.py")

# start streamlit
subprocess.Popen(f'python -m streamlit run "{app_path}"', shell=True)

# wait for server
time.sleep(6)

# open browser
webbrowser.open("http://localhost:8501")