import json
import numpy as np
from collections import Counter
from tqdm import tqdm
result_path = "/gallery_tate/jaehyuk.sung/tasks/vr/results.json"
with open(result_path, "r") as fp:
    vv = json.load(fp)

def gt_lab(labels) :
    return [[int(n) for n in tmp] for tmp in labels]

cnt = 0
tq = tqdm(vv)
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