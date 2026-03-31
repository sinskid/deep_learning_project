import torch

# Fonction de training pour un epoch 
# Avec évaluation sur le test set à la fin de l'epoch pour permettre l'early stopping

def train(model, train_dataloader, test_dataloader, optimizer, loss_function, device):

    # Train the model
    model.train()
    train_loss = 0

    for x, y in train_dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_function(outputs, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    # Evaluate the model
    model.eval()
    test_loss = 0
    
    for x, y in test_dataloader:
        x, y = x.to(device), y.to(device)
        
        outputs = model(x)
        loss = loss_function(outputs, y)
        
        test_loss += loss.item()

    return train_loss / len(train_dataloader), test_loss / len(test_dataloader)