import torch
from src import dataset, model, train, prune

def main():
    train_loader, val_loader, _ = dataset.load_datasets()
    net = model.SimpleCNN()

    # Training
    net = train.train_model(net, train_loader)
    accuracy = train.evaluate_model(net, val_loader)
    print(f"Validation Accuracy before pruning: {accuracy}%")

    # Pruning
    net = prune.prune_model(net)
    accuracy_after_pruning = train.evaluate_model(net, val_loader)
    print(f"Validation Accuracy after pruning: {accuracy_after_pruning}%")

    # Save models
    torch.save(net.state_dict(), './models/model.pth')
    torch.save(net.state_dict(), './models/pruned_model.pth')

if __name__ == "__main__":
    main()
