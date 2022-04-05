import numpy as np
import torch

import clover_utils.hubconf

# Check what entrypoints are available
entrypoints = torch.hub.list('jpl-clover/clover_utils', force_reload=True)


# Try loading a model and confirming it works by pushing through a tensor
# and conducting a forward pass
model = clover_utils.hubconf.finetuned_supervised_resnet()
# Confirm that this outputs logits over 19 classes
print(model(torch.zeros(1,3,227,227)).shape)