import torch
import torch.nn.utils.prune as prune

def prune_model(model, prune_ratio=0.5):
    prune.ln_structured(model.conv1, name="weight", amount=prune_ratio, n=2, dim=0)
    prune.ln_structured(model.conv2, name="weight", amount=prune_ratio, n=2, dim=0)
    
    prune.l1_unstructured(model.fc1, name='weight', amount=prune_ratio)
    prune.l1_unstructured(model.fc2, name='weight', amount=prune_ratio)
    
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.remove(module, 'weight')

    return model

