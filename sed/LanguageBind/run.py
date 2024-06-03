import pysrt
import datetime
import torchaudio
import os
import torch
import json
from datetime import timedelta
from languagebind import LanguageBindAudio, LanguageBindAudioTokenizer, LanguageBindAudioProcessor
import subprocess
from sklearn.metrics import precision_score, recall_score, f1_score

d = datetime.datetime(2000, 1, 1)
labels = []

def conv(dates) :
    return dates.strftime("%H:%M:%S:%f")[:-3]

def sec_cal(time) :
    if type(time) == str :
        time = datetime.datetime.strptime(time, "%H:%M:%S:%f").time()
    d = datetime.datetime(2000, 1, 1)
    return (datetime.datetime.combine(d, time) - d).total_seconds()

def cm(ans, pred) :
    cp1 = 1 if ans == 'True' else 0
    cp2 = 1 if pred == 'True' else 0
    if cp1==1:
        if cp1 == cp2:
            return "TP"
        return "FN"
    else :
        if cp1 == cp2:
            return "TN"
        return "FP"
    
if __name__ == "__main__" :
    with open('./config.json', "r") as fp :    
        config = json.load(fp)
    music_path = config['movie_path']
    save_path = config['save_path']
    sub_path = config['sub_path']
    nxt = timedelta(seconds=config["time_delta"])
    bd = timedelta(seconds=config["boundary_delta"])
    
    subs = pysrt.open(sub_path)
    
    for sub in subs:
        # hours, miniutes, seconds, milliseconds
        nw = sub.start.to_time()
        en = sub.end.to_time()
        while True:
            if nw == en:
                break
            st = datetime.datetime.combine(d, nw)
            if (st + bd).time() >= en :
                labels.append((conv(nw), conv(en), sub.text))
                nw = en
            else :
                labels.append((conv(nw), conv((st + nxt).time()), sub.text))
                nw = (st + nxt).time()
        # print(sub.text)
        # for nw in en
        ab = datetime.datetime.combine(d, sub.start.to_time())    
    
    # Music
    if os.path.splitext(music_path)[1] != '.wav' :        
        command = "ffmpeg -i {} -ab 160k -ac 2 -ar 44100 -vn {}".format(music_path, os.path.splitext(music_path)[0] + '.wav')
        subprocess.call(command, shell=True)
        music_path = os.path.splitext(music_path)[0] + '.wav'
        # print(music_path)
    audio, sample_rate = torchaudio.load(music_path)
    # Dataset
    split_result = []
    gt = []
    for i in labels :
        start = sec_cal(i[0])
        end = sec_cal(i[1])
        # print(start * sample_rate, end * sample_rate)
        split_result.append(audio[:, int(start * sample_rate) : int(end * sample_rate)])
        gt.append(i[2])
    
    print("=========Model Load=========")
    pretrained_ckpt = 'LanguageBind/LanguageBind_Audio'  # also 'LanguageBind/LanguageBind_Audio_FT'
    # checkpoint_path = args.checkpoint_path if args.checkpoint_path is not None else ""
    model = LanguageBindAudio.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir')
    tokenizer = LanguageBindAudioTokenizer.from_pretrained(pretrained_ckpt, cache_dir='./cache_dir')
    audio_process = LanguageBindAudioProcessor(model.config, tokenizer)
    model.eval()
    
    print("=========Data process & Infer=========")
    prompt = ["Only Voice", 'Background music']
    data = audio_process(split_result, prompt, return_tensors='pt')
    with torch.no_grad() :
        results = model(**data)
    score_matrix = results.image_embeds @ results.text_embeds.T
    score_matrix_c = torch.softmax(score_matrix, dim=-1)
    gt_l = [1 if i == 'True' else 0 for i in gt]
    final_predict = score_matrix_c.argmax(dim=-1).tolist()    
    
    prec = precision_score(final_predict, gt_l)
    rec = recall_score(final_predict, gt_l)
    f1 = f1_score(final_predict, gt_l)
    
    # Export result
    results = []
    # print(len(labels), len(gt_l))
    for idx, i in enumerate(labels) :
        start_time, end_time, gt_nw = i
        predict_nw = "True" if final_predict[idx] == 1 else "False"
        
        results.append({'start_time': start_time, 'end_time': end_time, 'ground_truth': gt_nw, 'predict': predict_nw, 'confusion matrix': cm(gt_nw, predict_nw)})
    
    metric = {'prec': prec, 'rec': rec, 'f1 score': f1}
    
    if not os.path.exists(save_path) :
        os.makedirs(save_path)
        
    with open(os.path.join(save_path, 'result.json'), 'w') as fp:
        json.dump(results, fp)
        fp.close()
        
    with open(os.path.join(save_path, 'metric.json'), 'w') as fp:
        json.dump(metric, fp)
        fp.close()
    # print(results)        