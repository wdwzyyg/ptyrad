import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
import numpy as np
from accelerate import Accelerator
dist.init_process_group(backend='gloo') # gloo backend is needed to run DDP on Windows

def cycle(it):
    while True:
        for el in it:
            yield el

# Define your BatchesDataset
class IndicesDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.indices[idx]

# Sample data for testing

len_indices = 16
batch_size = 4

indices = np.random.randint(0,32,len_indices)
dataset = IndicesDataset(indices)

# Create DataLoader with no batching since we want to test the splitting
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Initialize Accelerator
accelerator = Accelerator(split_batches = True)

# Prepare the model (dummy model here) and dataloader
model = torch.nn.Linear(1, 1)  # Dummy model
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

print(f"indices = {indices}")
# Test if batches are being split correctly
print(f"len(dataloader) = {len(dataloader)}, dataloader.batch_size = {len_indices//len(dataloader)}")
for batch_idx, batch in enumerate(dataloader):
    # This will show how the data is split across calls
    print(f"Batch idx = {batch_idx}, Batch value = ", batch)

# Note: When `split_batches=True`, the accelerator will automatically handle splitting for each process.
