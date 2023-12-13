import pandas as pd
import json, requests
import sys
import tqdm
from collections import defaultdict


all_results = defaultdict(int)
out_data = {}
results = []
with open(sys.argv[1], 'r') as f:
   data = json.load(f)
   for utt, gt_responses in tqdm.tqdm(data.items(), total=len(data)):
        r  = requests.post("http://localhost:8202", json={'q': utt})
        if r.status_code != 200:
            print("Error with request")
            continue
        response = r.json()
        predicted = response['wiki'][0]['url']
        results.append(predicted in gt_responses)
print(sum(results)/len(results))
