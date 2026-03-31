import timm
import torchvision.models as models
import torch.nn as nn
import torch
from python.data import vit_base_transform, cnn_base_transform

### MODELES ET FONCTIONS D'EVALUATIONS

def get_cnn(num_classes=10):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_vit(num_classes=10):
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model

def accuracy1(dataloader, model, device):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            total_correct += torch.sum(preds == labels).item()
            total_samples += labels.size(0)

    return total_correct / total_samples

# if the prediction is in top 2 predictions, it is considered correct
def accuracy2(dataloader, model, device, num_classes=10):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.topk(outputs, k=2, dim=1)

            for i in range(len(labels)):
                label = labels[i].item()
                pred = preds[i].tolist()
                if label in pred:
                    total_correct += 1
            total_samples += labels.size(0)

    return total_correct / total_samples

def accuracy_per_class1(dataloader, model, device, num_classes=10):
    model.eval()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            for i in range(len(labels)):
                label = labels[i].item()
                pred = preds[i].item()
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1

    class_accuracies = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(num_classes)]
    return class_accuracies

# if the prediction is in top 2 predictions, it is considered correct
def accuracy_per_class2(dataloader, model, device, num_classes=10):    
    model.eval()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.topk(outputs, k=2, dim=1)

            for i in range(len(labels)):
                label = labels[i].item()
                pred = preds[i].tolist()
                if label in pred:
                    class_correct[label] += 1
                class_total[label] += 1
    
    class_accuracies = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(num_classes)]
    return class_accuracies
