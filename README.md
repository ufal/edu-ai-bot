# qa_server

- communication with server: `r  = requests.post("server:port", json={'q': 'query'})`

## Run the server
### TODO

## Configuration
### TODO

## Train the classifier
To retrain the intent classifier, run the following command (default parameters, change optionally):
```shell
python -m educlf.train --data_fn data/merged_intents.tsv --model robeczech --out_dir $(pwd)/trained_classifierXY
```