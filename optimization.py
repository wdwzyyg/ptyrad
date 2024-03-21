## Loss, regularization, constraints
import torch
import torch.nn as nn
from torchmetrics.image import TotalVariation
# https://lightning.ai/docs/torchmetrics/stable/image/total_variation.html 
# This TV only applies to the last 2 dim (N,C,H,W)

class CombinedLoss(nn.Module):
    def __init__(self, loss_params, device='cuda:0'):
        super(CombinedLoss, self).__init__()
        self.device = device
        self.loss_params = loss_params
        self.mse = nn.MSELoss()
        self.tv = TotalVariation().to(device)

    def forward(self, model_CBEDs, measured_CBEDs, opt_obj):
        losses = []

        # Calculate loss_single
        single_params = self.loss_params['loss_single']
        if single_params['state']:
            dp_pow = single_params.get('dp_pow', 0.5)
            loss_single = self.mse(model_CBEDs.pow(dp_pow), measured_CBEDs.pow(dp_pow))**0.5
            loss_single *= single_params['weight']
        else:
            loss_single = torch.zeros(1, device=self.device) # Return a tensor 0 so that the append/sum would work normally without NaN
        losses.append(loss_single)

        # Calculate loss_pacbed
        pacbed_params = self.loss_params['loss_pacbed']
        if pacbed_params['state']:
            dp_pow = pacbed_params.get('dp_pow', 0.2)
            loss_pacbed = self.mse(model_CBEDs.mean(0).pow(dp_pow), measured_CBEDs.mean(0).pow(dp_pow))**0.5
            loss_pacbed *= pacbed_params['weight']
        else:
            loss_pacbed = torch.zeros(1, device=self.device)
        losses.append(loss_pacbed)

        # Calculate loss_tv
        tv_params = self.loss_params['loss_tv']
        if tv_params['state']:
            loss_tv = tv_params['weight'] * self.tv(opt_obj[..., 1])
        else:
            loss_tv = torch.zeros(1, device=self.device)
        losses.append(loss_tv)

        # Calculate loss_l1
        l1_params = self.loss_params['loss_l1']
        if l1_params['state']:
            loss_l1 = l1_params['weight'] * torch.mean(opt_obj[..., 1].abs())
        else:
            loss_l1 = torch.zeros(1, device=self.device)
        losses.append(loss_l1)

        # Calculate loss_l2
        l2_params = self.loss_params['loss_l2']
        if l2_params['state']:
            loss_l2 = l2_params['weight'] * torch.mean(opt_obj[..., 1].pow(2))
        else:
            loss_l2 = torch.zeros(1, device=self.device)
        losses.append(loss_l2)

        total_loss = sum(losses)
        return total_loss, losses