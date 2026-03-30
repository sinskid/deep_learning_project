import os

# Gere la création des dossiers pour les modèles et les logs 
# En fonction de l'environnement (local ou Colab)
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


# Fonction tenseur 3,H,W vers numpy H,W,3 pour affichage
def tensor_to_image(tensor):
    image = tensor.cpu().numpy().transpose(1, 2, 0)
    image = (image * 255).astype('uint8')
    return image

# Fonction pour afficher une image avec son label
def show_image(image, label):
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()