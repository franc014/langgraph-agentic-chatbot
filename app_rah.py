"""Streamlit entrypoint for the RAG application."""

from pathlib import Path
import runpy


# Streamlit reruns this file for every interaction. Running the implementation
# as a script ensures its UI statements execute on every rerun as well.
runpy.run_path(str(Path(__file__).with_name("app_rag.py")))
