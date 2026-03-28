from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import ImageDraw
import torch
import random

# --- Masque centré avec pourcentage ---
def apply_center_mask(image, percent=0.2):
    """
    Applique un carré noir centré sur l'image.
    percent : proportion du plus petit côté à masquer (0.0 - 1.0)
    """
    img = image.copy()
    w, h = img.size
    mask_size = int(min(w, h) * percent)

    # Coordonnées du carré centré
    x0 = (w - mask_size) // 2
    y0 = (h - mask_size) // 2
    x1 = x0 + mask_size
    y1 = y0 + mask_size

    draw = ImageDraw.Draw(img)
    draw.rectangle([x0, y0, x1, y1], fill=(0,0,0))
    return img

# --- Transformations de base ---
base_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor()
])

# --- Classe Dataset augmentée ---
class AugmentedDataset(Dataset):
    def __init__(self, hf_dataset, base_transform=None, effects=None):
        """
        hf_dataset : Hugging Face dataset
        effects : liste d'effets à appliquer ('mask', 'blur', 'rotate', 'color')
        base_transform : transformations de base (resize, crop, to tensor)
        effect_parameters : dictionnaire des paramètres pour chaque effet
        """
        self.dataset = hf_dataset
        self.effects = effects if effects is not None else {}
        self.base_transform = base_transform if base_transform is not None else T.ToTensor()
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        label = self.dataset[idx]['label']

        # --- Appliquer les effets ---
        for key, params in self.effects.items():
            if key == 'mask':
                image = apply_center_mask(image, percent=params)
            elif key == 'blur':
                image = image.filter(T.GaussianBlur(radius=params))
            elif key == 'color':
                factor = random.uniform(0.8, 1.2)
                image = T.functional.adjust_brightness(image, factor)
        
        # --- Transformations de base ---
        if self.base_transform:
            image = self.base_transform(image)

        return image, label

base_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor()
])


def load_data(batch_size=32,num_workers=2, effects = None, effects_parameters = None):
    
    dataset = load_dataset("blanchon/EuroSAT_RGB")
    
    dataset_train = AugmentedDataset(dataset['train'], base_transform=base_transform)
    dataset_test = AugmentedDataset(dataset['test'], base_transform=base_transform, effects=effects)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=num_workers)
    
    return train_loader, test_loader

