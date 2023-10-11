# EDU-AI-BOT

This is the main repo for the student's assistant bot. It consists of:
- intent classifier
- rule-based dialogue manager
- QA via Solr requests for background information + reranking + (optional) generative QA model
- generative chatbot (BlenderBot currently)

## Usage

- communication with server: `r  = requests.post("server:port", json={'q': 'query'})`

### Run the server

```
python api.py --port 8202 --logfile log.jsonl
```

### Keep the server running automatically

Edit your crontab (`crontab -e`) and enter this to add a check with auto-restart every 10 minutes:
```
*/10 * * * * bash /home/$USER/edu-ai-bot/scripts/autostart.sh <OPENAI_API_KEY> >/home/$USER/edu-ai-bot/cron.log 2>&1
```

## Configuration
### TODO


## Train the classifier
To retrain the intent classifier, run the following command (default parameters, change optionally):
```shell
python -m educlf.train --data_fn data/merged_intents.tsv --model robeczech --out_dir $(pwd)/trained_classifierXY
```

## Evaluate QA

```shell
python -m scripts.evaluate --eval_file data/edubot-data-augmented.json --data_root data/ --data_type edubot --config configs/default_config.yaml
```
