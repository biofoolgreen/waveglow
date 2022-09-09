# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import os
from scipy.io.wavfile import write
import torch
import time
import numpy as np
from mel2samp import files_to_list, MAX_WAV_VALUE
from denoiser import Denoiser


def main(num_runs, waveglow_path, is_fp16, sampling_rate, denoiser_strength, dynamic_mel_length=False):
    waveglow = torch.load(waveglow_path)['model']
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.cuda().eval()
    if is_fp16:
        from apex import amp
        waveglow, _ = amp.initialize(waveglow, [], opt_level="O3")

    if denoiser_strength > 0:
        denoiser = Denoiser(waveglow).cuda()
    dur_arr = []
    audio_arr = []
    rtf_arr = []
    mellen = 0
    mels = np.random.randint(100, 1300, size=num_runs)
    for i in range(num_runs):
        run_mel_length = mels[i] if dynamic_mel_length else 700
        mel = torch.rand((1, 80, run_mel_length)).cuda()
        mel = mel.half() if is_fp16 else mel
        start = time.time()
        with torch.no_grad():
            audio = waveglow.infer(mel, sigma=0.6)
            if denoiser_strength > 0:
                audio = denoiser(audio, denoiser_strength)
            audio = audio * MAX_WAV_VALUE
        dur = time.time() - start
        audio_duiration = audio.shape[1] / float(sampling_rate)
        rtf = audio_duiration / dur
        dur_arr.append(dur)
        audio_arr.append(audio_duiration)
        rtf_arr.append(rtf)
        mellen += mel.shape[-1]
        print(f"Mels shape: {mel.shape}\tAudio shape: {audio.shape}")
        print(f"Duration: {dur:.4f}s\tAudio duration: {audio_duiration:.2f}s\tRTF: {rtf:.2f}")
    print(f"[{len(dur_arr)}]Average duration: {np.mean(dur_arr[1:]):.4f}s")
    print(f"Average Audio duration: {np.mean(audio_arr):.4f}s")
    print(f"Average Mel length: {float(mellen/num_runs):.1f}s")
    print(f"Average RTF: {np.mean(rtf_arr[1:]):.4f}s")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_runs', default=100, type=int, 
                        help='Number of runs to get the benchmark.')
    parser.add_argument('-w', '--waveglow_path',
                        help='Path to waveglow decoder checkpoint with model')
    parser.add_argument("--sampling_rate", default=22050, type=int)
    parser.add_argument("--is_fp16", action="store_true")
    parser.add_argument("--is_dynamic_inputs", action="store_true")
    parser.add_argument("-d", "--denoiser_strength", default=0.0, type=float,
                        help='Removes model bias. Start with 0.1 and adjust')

    args = parser.parse_args()

    main(args.num_runs, args.waveglow_path, args.is_fp16, args.sampling_rate, args.denoiser_strength, args.is_dynamic_inputs)
