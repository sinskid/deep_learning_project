import os

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