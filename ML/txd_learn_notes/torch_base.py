import time

import numpy as np
import torch

x = torch.arange(12)
new_x = x.reshape(3, 4)

y = torch.randn(3, 4)
print(new_x.shape)
