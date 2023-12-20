import pandas as pd
import json, requests
import tqdm
import sys
from collections import defaultdict

all_results = defaultdict(int)
results_1 = []
results_3 = []
results_5 = []
results_detail = []
valid = 0
with open(sys.argv[1], 'r') as f:
    data = json.load(f)
    for n, (utt, correct) in enumerate(tqdm.tqdm(data.items(), total=len(data))):
        r  = requests.post("http://localhost:8202", json={'q': utt, 'debug': 1})
        if r.status_code != 200:
            print("Error with request")
            continue
        response = r.json()
        candidates = response['qa']
        candidates = [(c['content'].strip(' "').replace('""', '"'), c['score']) for c in candidates]
        valid += 1
        results_1.append(any([c[0] in correct for c in candidates[:1]]))
        results_3.append(any([c[0] in correct for c in candidates[:3]]))
        results_5.append(any([c[0] in correct for c in candidates[:5]]))
        results_detail.append((utt, candidates[0], correct))

print("Top-1: %d (%.4f)" % (sum(results_1), sum(results_1)/len(results_1)))
print("Top-3: %d (%.4f)" % (sum(results_3), sum(results_3)/len(results_3)))
print("Top-1: %d (%.4f)" % (sum(results_5), sum(results_5)/len(results_5)))

r1_filt = [r for (r, d) in zip(results_1, results_detail) if 'NENE' not in d[2]]
r3_filt = [r for (r, d) in zip(results_3, results_detail) if 'NENE' not in d[2]]
r5_filt = [r for (r, d) in zip(results_5, results_detail) if 'NENE' not in d[2]]
print("Valid:", len(r1_filt))
print("Top-1: %d (%.4f)" % (sum(r1_filt), sum(r1_filt)/len(r1_filt)))
print("Top-3: %d (%.4f)" % (sum(r3_filt), sum(r3_filt)/len(r3_filt)))
print("Top-1: %d (%.4f)" % (sum(r5_filt), sum(r5_filt)/len(r5_filt)))

print("%40.40s | %40.40s | %40.40s" % ('UTT', 'CAND', 'CORRECT'))
for r1, (utt, cand, corr) in zip(results_1, results_detail):
    if r1:
        continue
    if cand[:10] == corr[0][:10]:
        print(cand)
    print("%40.40s | %40.40s | %.4f | %40.40s" % (utt, cand[0], cand[1], corr[0]))
    for cor_it in corr[1:]:
        print(" " * 92 + (" | %40.40s" % cor_it))

