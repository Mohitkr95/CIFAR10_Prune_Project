import torch
import torch.nn as nn
import torch.optim as optim

def get_optimizer(model):
    return optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for _, (inputs, labels) in enumerate(train_loader, 0):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")
    return model

def evaluate_model(model, val_loader):
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total