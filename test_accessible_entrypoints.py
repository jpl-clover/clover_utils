import numpy as np
import torch

import hubconf

# Check what entrypoints/models are available in our hub conf
org_repo_branch = "jpl-clover/clover_utils:main"
entrypoints = torch.hub.list(org_repo_branch, force_reload=True)
print(f"Available entrypoints at '{org_repo_branch}'")
for entrypoint in entrypoints:
    print(f"\t{entrypoint}")