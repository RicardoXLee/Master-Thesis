import torch
from torch.distributions.dirichlet import Dirichlet

def uce_loss(alpha, target, regularization_weight=1e-5):
    """
    Uncertainty-weighted Cross Entropy Loss (UCE Loss).
    
    Parameters:
    - alpha: Dirichlet parameters predicted by the model (torch.Tensor).
    - target: One-hot encoded true labels (torch.Tensor).
    - regularization_weight: Weight for the entropy regularization term (float).
    
    Returns:
    - loss: Computed UCE Loss (torch.Tensor).
    """
    # Ensure numerical stability by adding a small constant
    epsilon = 1e-8
    alpha = alpha + epsilon

    # Calculate alpha_0 (sum of Dirichlet parameters)
    alpha_0 = torch.sum(alpha, dim=1, keepdim=True)

    # Calculate digamma differences
    digamma_alpha_0 = torch.digamma(alpha_0)
    digamma_alpha = torch.digamma(alpha)

    # Cross-entropy loss component
    uce = torch.sum(target * (digamma_alpha_0 - digamma_alpha), dim=1)

    # Regularization term: Entropy of the Dirichlet distribution
    entropy_reg = Dirichlet(alpha).entropy()

    # Combine loss and regularization
    loss = torch.mean(uce - regularization_weight * entropy_reg)

    return loss
