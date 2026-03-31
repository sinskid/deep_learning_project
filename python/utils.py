import os
import torch
# Gere la création des dossiers pour les modèles et les logs 

def setup_dirs():
    project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    paths = {
        "base": project_path,
        "models": os.path.join(project_path, "models"),
        "logs": os.path.join(project_path, "logs"),
        "images": os.path.join(project_path, "images")
    }

    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    return paths
