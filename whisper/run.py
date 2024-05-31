import os
import json
from os.path import isdir, join
from tqdm import tqdm
import whisper
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio
import numpy as np
import time
import torch

# Model Load
# processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium").to("cuda")
print("=======Model Load Start!=======")
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2").to("cuda")
forced_decoder_ids = processor.get_decoder_prompt_ids(language="korean", task="transcribe")
print("=======Model Load End!=======")


def check(list1, list2) :
    from collections import Counter
    if Counter(list1) == Counter(list2) :
        return 1
    else :
        return 0

def get_answer(filename):
    with open(filename, "r") as fp :
        vv = json.load(fp)
    return vv["전사정보"]["OrgLabelText"]
    
default_path = "./"
datasets = join(default_path, 'datasets')
# print(os.listdir(datasets))
already = [os.path.splitext(i)[0] for i in os.listdir(join(default_path, "TL11_large")) if i.endswith(".json")]

labeling_path = join(datasets, 'labeling')
labeling_sets = sorted([f for f in os.listdir(labeling_path) if isdir(join(labeling_path, f))])
source_path = join(datasets, 'source')
source_sets = sorted([f for f in os.listdir(source_path) if isdir(join(source_path, f))])
labeling_path1 = 'TL11'
# TODO: change path
# for labeling_path1 in labeling_sets :
numbering = 'TS' + labeling_path1[2:] 
la_p1 = join(labeling_path, labeling_path1, '1')
so_p1 = join(source_path, numbering, '1')
la_ls = os.listdir(la_p1)
for labeling_path2 in la_ls :
    la_p2 = join(la_p1, labeling_path2)
    so_p2 = join(so_p1, labeling_path2)
    la_l = os.listdir(la_p2)
    so_l = os.listdir(so_p2)
    for _, i in enumerate(tqdm(la_l)) :
        if i in already:
            print(f"already exist! {i}")
            continue
        la_p3 = join(la_p2, i)
        so_p3 = join(so_p2, i)
        json_write = []
        tmps = os.listdir(la_p3)
        files_name = [os.path.splitext(filename)[0] for filename in tmps]
        offset = 16
        batches = []
        for itr in range(0, len(files_name), offset) :
            batches.append(files_name[itr : itr+offset])
        for index, batch in enumerate(tqdm(batches)) :
            # batch: file_name
            answer_json = [get_answer(join(la_p3, name)) for name in tmps]
            wav_files = [filename + '.wav' for filename in batch]
            ds = load_dataset(so_p3, data_files = wav_files)["train"].cast_column("audio", Audio(sampling_rate = 16000))
            # print(ds)
            raw_audio = [x["array"].astype(np.float32) for x in ds["audio"]]
            input_features = processor(raw_audio, return_tensors="pt", sampling_rate = 16000).input_features 
            # print("=======Dataset Load End!=======")
            # generate token ids
            with torch.no_grad() :
                predicted_ids = model.generate(input_features.to("cuda"), forced_decoder_ids= forced_decoder_ids)
            # decode token ids to text
            # transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            predicts = [trans.strip() for trans in transcription]
                
            for file_name, answer, predict in zip(batch, answer_json, predicts) :
                json_write.append({
                    "file": file_name,
                    "gt": answer,
                    "predict": predict
                })
            if index % 10 == 0 :
                print(f"{index} / {len(batches)} ends\n")
        with open("/gallery_tate/jaehyuk.sung/tasks/whisper/results/TL11_large/" + i + ".json", "w") as fp :
            json.dump(json_write, fp)