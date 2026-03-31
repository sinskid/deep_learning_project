import os
import torch
import matplotlib.pyplot as plt
import numpy as np
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

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def show_nb_images_per_label(dataset, class_names, nb_images=2, save_path=None):
    
    label_images = defaultdict(list)
    num_labels = len(class_names)

    # Récupérer 2 images par label
    for i in range(len(dataset)):
        img, label = dataset[i]["image"], dataset[i]["label"]
        
        if len(label_images[label]) < nb_images:
            label_images[label].append(img)
        
        # Stop quand toutes les classes ont 2 images
        if all(len(imgs) == nb_images for imgs in label_images.values()) and len(label_images) == num_labels:
            break

    fig, axes = plt.subplots(nb_images, num_labels, figsize=(3*num_labels, 3*nb_images))
    
    # 🔥 Toujours forcer axes en 2D
    axes = np.array(axes).reshape(nb_images, num_labels)

    for i, (label, imgs) in enumerate(label_images.items()):
        for j, img in enumerate(imgs):
            ax = axes[j, i]
            ax.imshow(img)
            ax.axis('off')

            title = class_names[label]

            if j == 0:  # Afficher le titre seulement pour la première image de chaque label
                ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


labels_to_names = {
    0: "AnnualCrop",
    1: "Forest",
    2: "HerbaceousVegetation",
    3: "Highway",
    4: "Industrial",
    5: "Pasture",
    6: "PermanentCrop",
    7: "Residential",
    8: "River",
    9: "SeaLake"
}