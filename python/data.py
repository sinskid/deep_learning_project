from datasets import load_dataset
from matplotlib import image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import ImageDraw, Image, ImageFilter
import torch


# Transformations de base : image 224x224 pour vit et cnn , inconvenients -> étire les images 
# Parametres de normalisation différents pour ViT et CNN (ImageNet)
vit_base_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

])
cnn_base_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Classe Dataset augmentée : permet d'appliquer plusieurs effets pour mesurer la différence de performance entre les modèles ---
class AugmentedDataset(Dataset):
    def __init__(self, hf_dataset, base_transform=None, effects=None):
        """
        hf_dataset : Hugging Face dataset
        effects : liste d'effets à appliquer ('mask', 'blur', 'color')
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

        for key, param in self.effects.items():
            if key == 'center_mask':
                image = apply_center_mask(image, param)
            elif key == 'random_mask':
                image = apply_random_mask(image, param)
            elif key == 'gaussian_blur':
                image = apply_blur(image, param)
        image = self.base_transform(image)
        return image, label

def load_data(batch_size=32, num_workers=2, data_type=None, model_name = None, effects=None):
    
    # dataset Hugging Face 
    dataset = load_dataset("blanchon/EuroSAT_RGB")
    
    if model_name == "vit":
        base_transform = vit_base_transform
    elif model_name == "cnn":
        base_transform = cnn_base_transform
    else:
        base_transform = None
    
    # test dataset
    dataset = AugmentedDataset(dataset[data_type], base_transform=base_transform, effects=effects)

    # test dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    
    return dataloader

# --- Fonctions d'effets sur les images ---

# --- Masque centré sur PIL.Image ---
def apply_center_mask(image, percent):
    """
    image : PIL.Image
    percent : fraction du plus petit côté à masquer
    """
    img = image.copy()  # pour ne pas modifier l'original
    W, H = img.size
    mask_size = int(min(W, H) * percent)
    
    x0 = (W - mask_size) // 2
    y0 = (H - mask_size) // 2
    
    draw = ImageDraw.Draw(img)
    draw.rectangle([x0, y0, x0 + mask_size, y0 + mask_size], fill=(0, 0, 0))
    
    return img

# --- Masque aléatoire sur PIL.Image ---
def apply_random_mask(image, percent):
    """
    image : PIL.Image
    percent : fraction de pixels à masquer
    """
    img = image.copy()
    W, H = img.size
    
    # Créer un masque binaire aléatoire H x W
    mask = (torch.rand(H, W) > percent).numpy().astype('uint8') * 255
    
    # Convertir en image binaire et appliquer sur chaque canal
    mask_img = Image.fromarray(mask, mode='L')
    if img.mode != 'RGB':
        img = img.convert('RGB')
    r, g, b = img.split()
    r = Image.composite(r, mask_img.point(lambda x: 0), mask_img)
    g = Image.composite(g, mask_img.point(lambda x: 0), mask_img)
    b = Image.composite(b, mask_img.point(lambda x: 0), mask_img)
    
    return Image.merge('RGB', (r, g, b))

def apply_blur(image, radius):
    return image.filter(ImageFilter.GaussianBlur(radius))