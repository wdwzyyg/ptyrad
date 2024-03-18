## Loss, regularization, constraints

import torch

def cbed_rmse(y_pred, y_true):
    """
    Calculate the root mean squared error (RMSE) for PtychoShelves formulation in PyTorch.
    E = (modF - aPsi)^2 / (0.5)^2
    Err = sqrt(mean(E))
    
    Args:
        y_pred: Predicted values (modF, amplitude of the wave function on the detector plane)
        y_true: Ground truth values (aPsi, amplitude of the wave function on the detector plane)
        
    Returns:
        cbed_rmse: Root Mean Squared Error (RMSE)
    """
    mse_loss = torch.nn.MSELoss()
    cbed_rmse = torch.sqrt(mse_loss(y_pred, y_true))
    
    return cbed_rmse