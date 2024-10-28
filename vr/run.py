import json
import os
import numpy as np
from pyannote.audio import Pipeline
from collections import Counter
from tqdm import tqdm

def path_find(path) :
    for (root, _, files) in os.walk(path, topdown=False) :
        if len(files) > 0:
            for file_name in files:
                name, ext = os.path.splitext(file_name)
                if ext.lower() == '.wav' :
                    chk = name + '.TXT'
                    if chk in files :
                        results.append({
                            'wav_path' : os.path.join(root, file_name),
                            'txt_path' : os.path.join(root, chk)
                        })

def gt_lab(labels) :
    return [[int(n) for n in tmp] for tmp in labels]
                        
if __name__ == "__main__" :
    with open('./config.json', "r") as fp :    
        config = json.load(fp)
    default_path = config['path']

    # [{'wav_path' : , 'txt_path':}]
    results = []

    path_find(default_path)
    pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",
                                        use_auth_token=config['use_auth_token'])
    vad_result = []
    cnt = 0
    # print(results)
    for i in results :
        wavs = i['wav_path']
        txts = i['txt_path']
        f_name = os.path.split(i['wav_path'])[0]
        w = []
        t = []
        # Predict
        for wav_output in pipeline(wavs).get_timeline().support():
            w.append([int(wav_output.start * 16000), int(wav_output.end * 16000)])
        # Grount_truth
        with open(txts, "r") as fp:
            for s in fp.readlines() :
                tmp = s.strip().split(maxsplit=2)
                t.append([tmp[0], tmp[1]])
        vad_result.append({
            'predict': w,
            'gt': t,
            'path': f_name
        })
    # print(vad_result)
    cnt = 0
    tq = tqdm(vad_result)
    f1_total = 0
    for _, v in enumerate(tq):
        preds = v['predict']
        gts = gt_lab(v['gt'])
        m = gts[0][1]
        fp = 0
        rang = np.zeros(gts[0][1])
        for pred in preds:
            rang[pred[0] : min(m, pred[1])] = 1
            fp += max(0, pred[1] - m)
        c = Counter(rang)
        tp = c[1]
        fn = c[0]
        if len(preds) == 0 :
            continue
        cnt += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * 1/(1/precision + 1/recall)
        f1_total += f1
        tq.set_description(f"prec: {precision:.2f}, rec: {recall:.2f}, total f1: {f1_total:.2f} => Avg: {f1_total/cnt:.2f}")
        
    print(f"===Final Result:{f1_total / cnt}===")
    
    with open('results.json', "w") as fp2 :
        json.dump(vad_result, fp2)
        fp2.close()
        
    with open('final_result.json', 'w') as fp2 :
        fp2.write(f"Final Result is {f1_total / cnt}")
        fp2.close()
