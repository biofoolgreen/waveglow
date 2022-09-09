import torch
import poptorch
import numpy as np
from glow import WN


class TestWN(torch.nn.Module):
    def __init__(self, 
                n_in_channels=16, 
                n_mel_channels=80, 
                n_layers=4, 
                n_channels=16, 
                kernel_size=3, 
                n_group=8):
        super(TestWN, self).__init__()
        self.wn = WN(n_in_channels=n_group//2, 
                    n_mel_channels=n_mel_channels*n_group, 
                    n_layers=n_layers, 
                    n_channels=n_channels,
                    kernel_size=kernel_size)
        self.unsample = torch.nn.ConvTranspose1d(n_mel_channels,
                                    n_mel_channels,
                                    1024, stride=256)
        self.n_group = n_group

    def forward(self, feats):
        audio, spect = feats
        spect = self.unsample(spect) # [b, 80, (C_in-1)*stride - 2*padding + kernel+outputpadding(0)]
        if spect.size(2) > audio.size(1):
            spect = spect[:, :, :audio.size(1)]
        spect = spect.unfold(2,  self.n_group, self.n_group).permute(0, 2, 1, 3) # [b, Lout//n_group, 80, n_group]
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)
        audio = audio.unfold(1, self.n_group//2, self.n_group).permute(0, 2, 1)
        print(audio.shape, spect.shape)
        out = self.wn((audio, spect))
        return out


wn_cpu = TestWN()

wn = TestWN()
wn.load_state_dict(wn_cpu.state_dict())
wn = poptorch.inferenceModel(wn)

spect = torch.FloatTensor(batch_size, 80, 100).normal_()
audio = torch.FloatTensor(batch_size, 1498).normal_()

out = wn((audio, spect))
out_cpu = wn_cpu((audio, spect))

print(f"CPU: {out_cpu.shape}\tIPU: {out.shape}")
print(np.allclose(out.detach().numpy(), out_cpu.detach().numpy(), rtol=1e-15, atol=1e-15, equal_nan=False))