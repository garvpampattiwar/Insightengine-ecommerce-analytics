"""
Script to execute all 5 notebooks in place using nbclient/nbconvert
so that all code cells have evaluated outputs, logs, and plots.
"""

from pathlib import Path
import nbformat
from nbclient import NotebookClient

notebooks = [
    "01_data_cleaning.ipynb",
    "02_eda.ipynb",
    "03_segmentation.ipynb",
    "04_churn.ipynb",
    "05_recommender.ipynb",
]

for nb_name in notebooks:
    nb_path = Path("notebooks") / nb_name
    print(f"Executing {nb_name}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()

    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Successfully executed and saved {nb_name}.")

print("\nAll 5 notebooks executed successfully!")
