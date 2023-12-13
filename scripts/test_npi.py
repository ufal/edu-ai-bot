import pandas as pd
import json, requests
import tqdm
import sys
from collections import defaultdict

all_results = defaultdict(int)
results_1 = []
results_3 = []
results_5 = []
valid = 0
with open(sys.argv[1], 'r') as f:
    data = json.load(f)
    for n, (utt, correct) in enumerate(tqdm.tqdm(data.items(), total=len(data))):
        if 'NENE' in correct or len(correct) == 0 or len(correct[0]) == 0:
            continue
        r  = requests.post("http://localhost:8202", json={'q': utt})
        if r.status_code != 200:
            print("Error with request")
            continue
        response = r.json()
        candidates = response['qa']
        candidates = [c['content'].strip() for c in candidates]
        valid += 1
        results_1.append(any([c in correct for c in candidates[:1]]))
        results_3.append(any([c in correct for c in candidates[:3]]))
        results_5.append(any([c in correct for c in candidates[:5]]))

print("Valid:", valid)
print("Top-1", sum(results_1)/len(results_1))
print("Top-3", sum(results_3)/len(results_3))
print("Top-5", sum(results_5)/len(results_5))
