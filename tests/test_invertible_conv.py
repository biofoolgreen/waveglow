import numpy as np
import torch
import poptorch


class Invertible1x1Conv(torch.nn.Module):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """

    def __init__(self, c):
        super(Invertible1x1Conv, self).__init__()
        self.conv = torch.nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0,
                                    bias=False)

        # Sample a random orthonormal matrix to initialize weights
        W = torch.linalg.qr(torch.FloatTensor(c, c).normal_())[0]

        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:, 0] = -1 * W[:, 0]
        W = W.view(c, c, 1)
        W = W.contiguous()
        self.conv.weight.data = W

    def forward(self, z):
        # shape
        batch_size, group_size, n_of_groups = z.size()

        W = self.conv.weight.squeeze()

        # Forward computation
        log_det_W = batch_size * n_of_groups * torch.logdet(W.unsqueeze(0).float()).squeeze()   # `logdet` op not support in IPU.
        z = self.conv(z)
        return z, log_det_W


    def infer(self, z):
        # shape
        batch_size, group_size, n_of_groups = z.size()

        W = self.conv.weight.squeeze()

        if not hasattr(self, 'W_inverse'):
            # Reverse computation
            W_inverse = W.float().inverse()     # `inverse` op not support in IPU.
            W_inverse = Variable(W_inverse[..., None])
            if z.type() == 'torch.cuda.HalfTensor' or z.type() == 'torch.HalfTensor':
                W_inverse = W_inverse.half()
            self.W_inverse = W_inverse
        z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
        return z


chn_size = 10

inputs = torch.FloatTensor(1,10,16).normal_()
# opts = poptorch.Options()
# opts.Jit.traceModel(True)
InverConv = poptorch.inferenceModel(Invertible1x1Conv(chn_size))

out = InverConv(inputs)
print(out)
