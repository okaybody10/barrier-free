import os
import json
import glob
import torch
import numpy as np
import os.path as path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio
from tqdm import tqdm
from metric import calc_result

# Model Load
# processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium").to("cuda")
"""
Extracts and returns the value associated with the key "전사정보" -> "OrgLabelText" from a JSON file.

Args:
    filename (str): The path to the JSON file to extract the value from.

Returns:
    str: The value associated with the key "전사정보" -> "OrgLabelText" in the JSON file.
"""
def get_answer(filename):
    with open(filename, "r") as fp :
        vv = json.load(fp)
    return vv["전사정보"]["OrgLabelText"]

"""
Creates a dictionary of files grouped by parent folders and a mapping of file names to paths.

Args:
    file_path (str): The path to the directory containing files.
    extension (str): The file extension to filter files by.

Returns:
    tuple: A tuple containing two dictionaries:
        - The first dictionary organizes files by parent folders.
        - The second dictionary maps file names to their full paths.
"""
def file_list(file_path, extension) :
    answer = {}
    get_path = {}
    for file in glob.iglob(f"{path.normpath(file_path)}/**/*.{extension}", recursive=True) :
    # print(file)
        paths, name_ext = path.split(file)
        name, _ = name_ext.split('.')
        paths = path.basename(paths)
        if paths in list(answer.keys()) :
            answer[paths].append(name)
        else :
            answer[paths] = [name]
        # answer.append({"folder": path.basename(paths), "name": name[:-5]})
        get_path[name] = file
    return answer, get_path

if __name__ == "__main__" :
    convert = {
        'tiny': 'openai/whisper-tiny',
        'base': 'openai/whisper-base',
        'small': 'openai/whisper-small',
        'medium': 'openai/whisper-medium',
        'large': 'openai/whisper-large',
        'large-v2': 'openai/whisper-large-v2'
    }
    with open("./config.json", "r") as fp :
        configs = json.load(fp)
    # Model Load
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=======Model Load Start!=======")
    
    try :
        model_size = convert[configs['model_size']]
    except KeyError:
        print("Incorret model size, please check")
        
    processor = WhisperProcessor.from_pretrained(model_size)
    model = WhisperForConditionalGeneration.from_pretrained(model_size).to(device)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="korean", task="transcribe")
    
    print("=======Model Load End!=======")
    print("=======Predict Start!=======")
    label_path = configs['labeling_path']
    source_path = configs['sound_path']
    save_path = configs['save_path']
    labeling, label_fpath = file_list(label_path, "json")
    source, source_fpath = file_list(source_path, "wav")
    final = {a: list(set(labeling[a]) & set(source[a])) for a in labeling.keys() & source.keys()}
    
    for key, value in final.items() :
        json_write = []
        offset = configs['batch']
        batches = []
        for itr in range(0, len(value), offset) :
            batches.append(value[itr : itr+offset])
        for index, batch in enumerate(tqdm(batches)) :
            answer_json = [get_answer(label_fpath[lab_path]) for lab_path in batch]
            wav_files = [src_name + '.wav' for src_name in batch]
            paths = path.dirname(source_fpath[batch[0]])
            ds = load_dataset(paths, data_files = wav_files)["train"].cast_column("audio", Audio(sampling_rate = 16000))
            raw_audio = [x["array"].astype(np.float32) for x in ds["audio"]]
            input_features = processor(raw_audio, return_tensors="pt", sampling_rate = 16000).input_features 
            # print("=======Dataset Load End!=======")
            # generate token ids
            with torch.no_grad() :
                predicted_ids = model.generate(input_features.to(device), forced_decoder_ids= forced_decoder_ids)
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
            if not os.path.exists(save_path) :
                os.makedirs(save_path)
        with open(os.path.join(save_path, key + ".json"), "w") as fp :
            json.dump(json_write, fp)
            fp.close()
    print("=======Predict End!=======")
    print("=======Performance calculate Start!=======")
    calc_result(save_path)
    print("=======Performance calculate End!=======")