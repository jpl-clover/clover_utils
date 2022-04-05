import numpy as np
import torch

import hubconf

# Try loading a model and confirming it works by pushing through a tensor
# and conducting a forward pass
# Here's how we can load our model
model = torch.hub.load('jpl-clover/clover_utils:main', 'finetuned_supervised_resnet', force_reload=True)
# Confirm that this outputs logits over 19 classes
in_shape = (1,3,227,227)
print(f"Inputting tensor of shape {in_shape} to downloaded model, result is {model(torch.zeros(*in_shape)).shape}")