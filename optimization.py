## Defining the loss function class, with loss and regularizations

from torchmetrics.image import TotalVariation
import torch

# This is a current working version (2024.03.23) of the CombinedLoss class
# I cleaned up the archived versions and slightly renamed the objects and variables for clarity

# The CombinedLoss takes a user-defined dict of loss_params, which specifies the state, weight, and param of each loss term
# Currently there're loss_single, loss_pacbed, loss_tv, loss_l1, and loss_l2 available
# The CBED related loss takes a parameter of dp_pow which raise the CBED with certain power, 
# usually 0.5 for loss_single and 0.2 for loss_pacbed to emphasize the diffuse background
# The obj-dependent regularizations (TV, L1, L2) are using the objp_patches as input
# In this way it'll only calculate values within the ROI, so the edges of the object would not be included

class CombinedLoss(torch.nn.Module):
    """ Calculate the loss with regularization on the object phase patches for each batch """
    
    # Example usage:
    # with torch.no_grad():
    #     loss_fn = CombinedLoss(loss_params, device=DEVICE)
    #     np.random.seed(42)
    #     indices = np.random.randint(0,N_max,48)
    #     model_CBEDs, objp_patches = model(indices)
    #     measured_CBEDs = measurements[indices]
    #     loss_batch, losses = loss_fn(model_CBEDs, measured_CBEDs, objp_patches, model.omode_occu)
    # print(losses)
    
    # model/measured CBEDs (N,Ky,Kx), float32 tensor
    # objp_patches (N, omode, Nz, Ny, Nx), float32 tensor
    # omode_occu (omode), float32 tensor
    
    def __init__(self, loss_params, device='cuda:0'):
        super(CombinedLoss, self).__init__()
        self.device = device
        self.loss_params = loss_params
        self.mse = torch.nn.MSELoss()
        self.tv = TotalVariation(reduction='mean').to(device)

    def forward(self, model_CBEDs, measured_CBEDs, objp_patches, omode_occu):
        losses = []

        # Calculate loss_single
        single_params = self.loss_params['loss_single']
        if single_params['state']:
            dp_pow = single_params.get('dp_pow', 0.5)
            loss_single = self.mse(model_CBEDs.pow(dp_pow), measured_CBEDs.pow(dp_pow))**0.5
            loss_single *= single_params['weight']
        else:
            loss_single = torch.tensor(0, device=self.device) # Return a scalar 0 tensor so that the append/sum would work normally without NaN
        losses.append(loss_single)

        # Calculate loss_pacbed
        pacbed_params = self.loss_params['loss_pacbed']
        if pacbed_params['state']:
            dp_pow = pacbed_params.get('dp_pow', 0.2)
            loss_pacbed = self.mse(model_CBEDs.mean(0).pow(dp_pow), measured_CBEDs.mean(0).pow(dp_pow))**0.5
            loss_pacbed *= pacbed_params['weight']
        else:
            loss_pacbed = torch.tensor(0, device=self.device)
        losses.append(loss_pacbed)

        # Calculate objp_patches_sum if needed
        if any(self.loss_params[loss_name]['state'] for loss_name in ['loss_tv', 'loss_l1', 'loss_l2']):
            objp_patches_sum = torch.sum(objp_patches * omode_occu[:, None, None, None], dim=1) # objp_patches_sum (N,Nz,Ny,Nx)
        
        # Calculate loss_tv
        # https://lightning.ai/docs/torchmetrics/stable/image/total_variation.html 
        # This TV only applies to the last 2 dim (N,C,H,W), so there's no z-direction measurement in our case with objp_patches (N, omode, Nz, Ny, Nx)
        tv_params = self.loss_params['loss_tv']
        if tv_params['state']:
            loss_tv = tv_params['weight'] * self.tv(objp_patches_sum)
        else:
            loss_tv = torch.tensor(0, device=self.device)
        losses.append(loss_tv)

        # Calculate loss_l1
        l1_params = self.loss_params['loss_l1']
        if l1_params['state']:
            loss_l1 = l1_params['weight'] * torch.mean(objp_patches_sum.abs())
        else:
            loss_l1 = torch.tensor(0, device=self.device)
        losses.append(loss_l1)

        # Calculate loss_l2
        l2_params = self.loss_params['loss_l2']
        if l2_params['state']:
            loss_l2 = l2_params['weight'] * torch.mean(objp_patches_sum.pow(2))
        else:
            loss_l2 = torch.tensor(0, device=self.device)
        losses.append(loss_l2)

        total_loss = sum(losses)
        return total_loss, losses
