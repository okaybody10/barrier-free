import json
from nlptutti import get_cer, get_wer, get_crr
from tqdm import tqdm
import os
# import nlptutti as metrics)
def in_folder(folder_path) :
    pass
result_def_path = "/gallery_tate/jaehyuk.sung/tasks/whisper/results/"
result_dir = os.listdir(result_def_path)
flag = True
for i in result_dir :
    cer = 0
    wer = 0
    crr = 0
    cnt = 0
    t_p = os.path.join(result_def_path, i)
    print(f"now directory: {t_p}")
    jsons = os.listdir(t_p)
    for f in jsons:
        file_p = os.path.join(t_p, f)
        with open(file_p, "r") as fp:
            vv = json.load(fp)
            # if flag is True :
            #     print(vv)
            #     flag = False
            tq = tqdm(vv)
            this_cnt = 0
            this_cer = 0
            this_wer = 0
            this_crr = 0
            for _, element in enumerate(tq) :
                gt = element['gt']
                pr = element['predict']
                cnt += 1
                this_cnt += 1
                n_cer = get_cer(gt, pr)['cer'] 
                cer += n_cer
                this_cer += n_cer
                n_wer =get_wer(gt, pr)['wer'] 
                wer += n_wer
                this_wer += n_wer
                n_crr = get_crr(gt, pr)['crr']
                crr += n_crr
                this_crr += n_crr
                tq.set_description(f"Now cnt: {this_cnt}, cer: {this_cer/this_cnt:.2f}, wer: {this_wer/this_cnt:.2f}, crr: {this_crr/this_cnt:.2f}")
                # print(f"Now cnt: {cnt}, cer: {cer}")
    print(f"=====Final Result=====\ntotal: {cnt}\ncer: {cer/cnt}\nwer: {wer/cnt}\ncrr: {crr/cnt}")