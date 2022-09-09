import torch
import json
from mel2samp import Mel2Samp, load_wav_to_torch


filepath = r"/localdata/cn-customer-engineering/liguoying/datasets/LJSpeech-1.1/wavs/LJ050-0269.wav"
audio, sr = load_wav_to_torch(filepath)

with open("config.json") as f:
    data = f.read()
data_config = json.loads(data)["data_config"]
mel2samp = Mel2Samp(**data_config)
melspectrogram = mel2samp.get_mel(audio)

feats = [melspectrogram.unsqueeze(0), audio.unsqueeze(0)]
print([k.shape for k in feats])
torch_model = torch.load("checkpoints/waveglow_256channels_universal_v5.pt")["model"] # pytorch模型加载
# set the model to inference mode
torch_model.eval()

# export_onnx_file = "waveglow.onnx"					# 目的ONNX文件名
# torch.onnx.export(torch_model,
#                     feats,
#                     export_onnx_file,
#                     opset_version=11,
#                     do_constant_folding=True,	# 是否执行常量折叠优化
#                     input_names=["melspec", "audio"],		# 输入名
#                     output_names=["out_audio", "log_s", "log_det_W"],	# 输出名
#                     verbose=True, 
# )