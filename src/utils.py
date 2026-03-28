import os

def get_base_path():
    if "COLAB_GPU" in os.environ:
        return "/content/drive/MyDrive/deep_learning_project"
    return "./outputs"

def setup_dirs():
    base = get_base_path()

    paths = {
        "base": base,
        "models": os.path.join(base, "models"),
        "logs": os.path.join(base, "logs"),
    }

    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    return paths