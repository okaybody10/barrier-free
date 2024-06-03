import json
import glob
import os.path as path
from nlptutti import get_cer, get_wer, get_crr
from tqdm import tqdm
"""
Calculates and prints the average metrics (CER, WER, CRR) for ground truth and predicted values in JSON files.

Args:
    paths (str): The path to the directory containing JSON files.

Returns:
    None
"""
def calc_result(paths) :
    result_files = glob.glob(f"{path.normpath(paths)}/**/*.json", recursive=True)
    cer, wer, crr, cnt = [0] * 4
    for i in result_files :
        with open(i, "r") as fp:
            sheets = json.load(fp)
            tq = tqdm(sheets)
            this_cnt, this_cer, this_wer, this_crr = [0] * 4
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
        fp.close()
    print(f"=====Final Result=====\ntotal: {cnt}\ncer: {cer/cnt}\nwer: {wer/cnt}\ncrr: {crr/cnt}")
        
if __name__ == "__main__" :
    with open('./config.json', "r") as fp :
        tmp = json.load(fp)
        result_path = tmp['save_path']
        fp.close()
    calc_result(result_path)
    # Get all jsons