import torch
import poptorch
import numpy as np
from denoiser import Denoiser



chn_size = 10

inputs = torch.FloatTensor(1,10,16).normal_()
# opts = poptorch.Options()
# opts.Jit.traceModel(True)
InverConv = poptorch.inferenceModel(Invertible1x1Conv(chn_size))

out = InverConv(inputs)
print(out)
