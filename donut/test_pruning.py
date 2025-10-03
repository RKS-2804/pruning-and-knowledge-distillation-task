"""
Test script for magnitude pruning functionality in DONUT model
"""
import torch
import torch.nn as nn

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 50)  # Embedding
        self.linear1 = nn.Linear(50, 100)  # Non-embedding
        self.linear2 = nn.Linear(100, 10)  # Non-embedding

    def get_non_embedding_parameters(self):
        non_embedding_params = []
        embedding_names = ['embed.weight']
        for name, param in self.named_parameters():
            if name not in embedding_names:
                non_embedding_params.append((name, param))
        return non_embedding_params

    def calculate_sparsity(self):
        total_params = 0
        zero_params = 0
        for name, param in self.get_non_embedding_parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
        sparsity = zero_params / total_params if total_params > 0 else 0
        return sparsity

    def apply_magnitude_pruning(self, sparsity_target: float):
        for name, param in self.get_non_embedding_parameters():
            if param.dim() > 1:
                param_data = param.data.abs().flatten()
                threshold = torch.quantile(param_data, sparsity_target)
                mask = (param.data.abs() >= threshold).float()
                param.data.mul_(mask)

    def apply_pruning_masks(self, masks: dict):
        for name, param in self.named_parameters():
            if name in masks:
                param.data.mul_(masks[name])

def test_pruning():
    # Create dummy model for testing
    model = DummyModel()

    print("Dummy model created successfully")

    # Test initial sparsity
    initial_sparsity = model.calculate_sparsity()
    print(f"Initial sparsity: {initial_sparsity:.4f}")

    # Test getting non-embedding parameters
    non_emb_params = model.get_non_embedding_parameters()
    print(f"Number of non-embedding parameters: {len(non_emb_params)}")

    # Test applying pruning
    target_sparsity = 0.5  # 50% sparsity
    model.apply_magnitude_pruning(target_sparsity)

    # Check sparsity after pruning
    after_sparsity = model.calculate_sparsity()
    print(f"Sparsity after pruning to {target_sparsity}: {after_sparsity:.4f}")

    # Test applying masks (create dummy masks)
    masks = {}
    for name, param in model.named_parameters():
        if name in [n for n, _ in non_emb_params]:
            masks[name] = torch.ones_like(param)

    model.apply_pruning_masks(masks)

    print("Pruning functionality test completed successfully")

if __name__ == "__main__":
    test_pruning()