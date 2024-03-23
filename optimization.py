## Loss, regularization, constraints

import torch
from torchmetrics.image import TotalVariation
# https://lightning.ai/docs/torchmetrics/stable/image/total_variation.html 
# This TV only applies to the last 2 dim (N,C,H,W), so there's no z-direction measurement in our case with objp_patches (N, omode, Nz, Ny, Nx)


# Object patch regularization, this is the current working version as of 2024.03.23
# This version calculates the obj-dependent regularizations (TV, L1, L2) using the objp_patches
# In this way it'll only calculate values within the ROI, so the edges of the object would not be incorrectly included
class CombinedLoss_patch(torch.nn.Module):
    """ Calculate the loss with regularization on the object phase patches for each batch """
    
    # Example usage:
    # with torch.no_grad():
    #     loss_fn = CombinedLoss_patch(loss_params, device=DEVICE)
    #     np.random.seed(42)
    #     indices = np.random.randint(0,N_max,48)
    #     model_CBEDs, objp_patches = model(indices)
    #     measured_CBEDs = measurements[indices]
    #     loss_batch, losses = loss_fn(model_CBEDs, measured_CBEDs, objp_patches, model.omode_occu)
    # losses
    
    # model/measured CBEDs (N,Ky,Kx)
    # objp_patches (N, omode, Nz, Ny, Nx), float32 tensor
    # omode_occu (omode), float32 tensor
    
    def __init__(self, loss_params, device='cuda:0'):
        super(CombinedLoss_patch, self).__init__()
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
    
# Whole object regularization, archived on 2024.03.23
# This version is less ideal becasue the TV, L1, and L2 will be calculated based on the whole object
# In other words, it'll include the boundary region that's not being illuminated/optimized at all so
# calculating the loss with those regions included is a bit biased
class CombinedLoss_whole(torch.nn.Module):
    """ Calculate the loss with regularization on the entire object phase """
    # Example usage:
    # with torch.no_grad():
    #     loss_fn = CombinedLoss_whole(loss_params, device=DEVICE)
    #     np.random.seed(42)
    #     indices = np.random.randint(0,N_max,48)
    #     model_CBEDs, _ = model(indices)
    #     measured_CBEDs = measurements[indices]
    #     loss_batch, losses = loss_fn(model_CBEDs, measured_CBEDs, model.opt_objp, model.omode_occu)
    # losses

    # model/measured CBEDs (N,Ky,Kx)
    # opt_objp (omode, Nz, Ny, Nx), float32 tensor
    # omode_occu (omode), float32 tensor

    def __init__(self, loss_params, device='cuda:0'):
        super(CombinedLoss_whole, self).__init__()
        self.device = device
        self.loss_params = loss_params
        self.mse = torch.nn.MSELoss()
        self.tv = TotalVariation().to(device)

    def forward(self, model_CBEDs, measured_CBEDs, opt_objp, omode_occu):
        losses = []

        # Calculate loss_single
        single_params = self.loss_params['loss_single']
        if single_params['state']:
            dp_pow = single_params.get('dp_pow', 0.5)
            loss_single = self.mse(model_CBEDs.pow(dp_pow), measured_CBEDs.pow(dp_pow))**0.5
            loss_single *= single_params['weight']
        else:
            loss_single = torch.tensor(0, device=self.device) # Return a tensor 0 so that the append/sum would work normally without NaN
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

        # Calculate loss_tv
        tv_params = self.loss_params['loss_tv']
        if tv_params['state']:
            # Note that TV requires (N,C,Y,X) and opt_objp is (Nz, Ny, Nx) after reducing omode, so we'll add a singleton for N.
            loss_tv = tv_params['weight'] * self.tv(torch.sum(opt_objp * omode_occu[:, None, None, None], dim=0)[None,]) 
        else:
            loss_tv = torch.tensor(0, device=self.device)
        losses.append(loss_tv)

        # Calculate loss_l1
        l1_params = self.loss_params['loss_l1']
        if l1_params['state']:
            loss_l1 = l1_params['weight'] * torch.mean(torch.sum(opt_objp * omode_occu[:, None, None, None], dim=0).abs())
        else:
            loss_l1 = torch.tensor(0, device=self.device)
        losses.append(loss_l1)

        # Calculate loss_l2
        l2_params = self.loss_params['loss_l2']
        if l2_params['state']:
            loss_l2 = l2_params['weight'] * torch.mean(torch.sum(opt_objp * omode_occu[:, None, None, None], dim=0).pow(2))
        else:
            loss_l2 = torch.tensor(0, device=self.device)
        losses.append(loss_l2)

        total_loss = sum(losses)
        return total_loss, losses