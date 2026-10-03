@echo off
rem Start the prototype demo. Double-click this file, or run it from any terminal.
rem Keep this window open while you use the demo; close it to stop the app.
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Could not find .venv\Scripts\python.exe in %CD%
  echo Create the environment first, see prototype\README.md
  pause
  exit /b 1
)
echo Starting the demo at http://localhost:8501 ...
echo Keep this window open while you use the demo. Close it to stop the app.
rem Open the browser a few seconds after the server starts.
start "" /b cmd /c "ping -n 6 127.0.0.1 >nul & start "" http://localhost:8501"
rem --server.headless true skips Streamlit's first-run e-mail question, which otherwise blocks start-up.
".venv\Scripts\python.exe" -m streamlit run prototype\app.py --server.headless true --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
pause
