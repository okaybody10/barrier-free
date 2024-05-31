# import torch
# import whisper
# import torchaudio 
# from transformers import WhisperForConditionalGeneration, AutoProcessor

# model = whisper.load_model("large")
# print("load finsish")
data_path = "/gallery_tate/jaehyuk.sung/tasks/whisper/datasets/source/TS1/1/1/0005_G1A3E7_KYG/"
vv = [
    "0005_G1A3E7_KYG_000004.wav",
    "0005_G1A3E7_KYG_000005.wav",
    "0005_G1A3E7_KYG_000006.wav",
    "0005_G1A3E7_KYG_000007.wav",
    "0005_G1A3E7_KYG_000008.wav",
    "0005_G1A3E7_KYG_000008.wav",
    "0005_G1A3E7_KYG_000009.wav",
    "0005_G1A3E7_KYG_000010.wav",
    "0005_G1A3E7_KYG_000011.wav",
    "0005_G1A3E7_KYG_000012.wav",
    "0005_G1A3E7_KYG_000013.wav",
    "0005_G1A3E7_KYG_000014.wav",
    "0005_G1A3E7_KYG_000015.wav",
    "0005_G1A3E7_KYG_000016.wav",
    "0005_G1A3E7_KYG_000017.wav",
    "0005_G1A3E7_KYG_000018.wav",
    "0005_G1A3E7_KYG_000019.wav",
    "0005_G1A3E7_KYG_000020.wav",
    "0005_G1A3E7_KYG_000021.wav",
    "0005_G1A3E7_KYG_000022.wav",
    "0005_G1A3E7_KYG_000023.wav",
    "0005_G1A3E7_KYG_000024.wav",
    "0005_G1A3E7_KYG_000025.wav",
    "0005_G1A3E7_KYG_000026.wav",
    "0005_G1A3E7_KYG_000028.wav",
    "0005_G1A3E7_KYG_000029.wav",
    "0005_G1A3E7_KYG_000030.wav",
    "0005_G1A3E7_KYG_000031.wav",
    "0005_G1A3E7_KYG_000032.wav",
    "0005_G1A3E7_KYG_000033.wav",
    "0005_G1A3E7_KYG_000034.wav",
    "0005_G1A3E7_KYG_000035.wav"
]
# # processor = AutoProcessor.from_pretrained("openai/whisper-large")
# # inputs = processor()
# # result = model.transcribe(vv, language='kr')
# # print(result['text'])

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio
import numpy as np
import time
import torch

print(torch.cuda.is_available())
# load model and processor
# processor = WhisperProcessor.from_pretrained("openai/whisper-small")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
start = time.time()
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2").to("cuda")
forced_decoder_ids = processor.get_decoder_prompt_ids(language="korean", task="transcribe")
end = time.time()
print(f"Model load End!, time: {end-start:.5f} sec")

# load dummy dataset and read audio files
start = time.time()
ds = load_dataset(data_path, data_files = vv)["train"].cast_column("audio", Audio(sampling_rate = 16000))
# print(ds)
raw_audio = [x["array"].astype(np.float32) for x in ds["audio"]]
end = time.time()
print(f"Dataload End!, time: {end-start:.5f} sec")
# print(raw_audio)
start = time.time()
input_features = processor(raw_audio, return_tensors="pt", sampling_rate = 16000).input_features 

# generate token ids
with torch.no_grad() :
    predicted_ids = model.generate(input_features.to("cuda"), forced_decoder_ids= forced_decoder_ids)
# decode token ids to text
# transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription)
end = time.time()
print(f"Transcribe End!, time: {end-start:.5f} sec")