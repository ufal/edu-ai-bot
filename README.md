# EDU-AI-BOT

This is the main repo for the student's assistant bot. It consists of:
- trainable intent classifier (based on the RobeCzech BERT-style model)
- rule-based dialogue manager (responses simply based on the intents)
- QA via Solr requests for background information + reranking + (optional) generative model to rephrase the answer
- chitchat chatbot: AIML or a generative chatbot (BlenderBot/DialoGPT)

The QA system is capable of querying Wikipedia as well as an in-domain database.
Which is queried is decided based on intent.

## Installation

Running the whole system assumes a recent Ubuntu (e.g. 22.04) and a corresponding recent Python (3.9/3.10),
running inside [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/). 
You can use the system-wide Python and Virtualenv instead, but you need to adapt some of the 
commands or scripts.

### Basic 

Create a conda environment:
```
conda create --name edubot python=3.10
```

Always activate the conda environment in order to run anything related to the bot:
```
conda activate edubot
```

Then clone this repository:
```
git clone https://github.com/ufal/edu-ai-bot.git
```

Then enter the `edu-ai-bot` directory, the following instructions assume running from that directory.

Install the required libraries:
```
pip install -r requirements.txt
```

### Installing the underlying Solr QA

You need to have `docker` and `docker-compose` installed (system-wide). 
First, create the directory `/var/edubot/qa-solr`, then unzip the contents of [`qa-solr/var.edubot.qa-solr.zip`](qa-solr/var.edubot.qa-solr.zip) 
in there. This includes the required Solr settings.

Then simply start up the dockerized solr instance by running:
```
sudo docker-compose -f qa-solr/docker-compose.yml up -d
```

### Installing data & models

All models should install automatically. Most of the used models are directly hosted on Huggingface,
the intent model is hosted [on the UFAL server](http://quest.ms.mff.cuni.cz/namuddis/intent-model.zip).

The QA Solr needs to be populated with data before use. 
You can download the data [from the UFAL server](http://quest.ms.mff.cuni.cz/namuddis/ir-data.tsv)
or prepare your own version (see "Updating" below).

To feed the data into Solr (assuming the Solr docker instance is up and running), run:
```
curl "http://localhost:8984/solr/wiki_test/update?commit=true&separator=%09&header=true&encapsulator=%C2%B6" --data-binary @ir_data.tsv -H 'Content-type:application/csv'
```


## Usage

### Running the server

To run the system in the default configuration, use this command:
```
OPENAI_API_KEY=<OPENAI_API_KEY> python api.py --port 8202 --logfile log.jsonl --config configs/default_config.yaml
```
You need to supply an OpenAI API key as an environment variable to be able to use OpenAI models.
The `--port` variable specifies the port on which the system will run, `--logfile` specifies the logfile and
`--config` specifies the configuration file to be loaded (see "Configuration" below for details).

### Querying the server

To communicate with the server, use the following request format (JSON):
```
{
    "q": "string",  # this is the user input for the system
    "conversation_id": "string" , # optional -- unique ID of the conversation, for better context-aware behavior
    "debug": 1,  # optional -- if this is supplied, more detailed responses are provided
    "exact": 1   # optional -- if this is supplied, a QA model will be applied on to Wiki QA results (otherwise raw best reranked Solr search result is returned)
    "site": "string"  # optional -- this can change QA in-domain behavior, depending on configuration (see below)
}
```

The response will be a JSON with the following fields:
```
{
    "a": "string",  # the main chatbot response
    "control": 1,  # will be returned for intents starting with "#" (suggesting control intents)
    "intent": [{"label": "string", "score": 0.98}, ...],  # debug only -- intent probabilities
    "wiki": [...], # debug only + only if Wiki is queried -- detailed QA results with cosine similarity scores
    "qa": [...], # debug only + only if in-doamin QA is queried -- detailed QA results with cosine similarity scores
}
```

### Keep the server running automatically

First, update your email and the Solr instance URL in the [`scripts/autostart.sh`](scripts/autostart.sh) file.
Then edit your crontab (`crontab -e`) and enter this to add a check with auto-restart every 10 minutes:
```
*/10 * * * * bash /home/$USER/edu-ai-bot/scripts/autostart.sh <OPENAI_API_KEY> >/home/$USER/edu-ai-bot/cron.log 2>&1
```

That script won't restart the Solr instance after reboot, it'll just send an email. 
In order to have Solr start automatically, the `root` user needs to have this in their crontab:
```
@reboot /home/$USER/edu-ai-bot/qa-solr/run.sh
```

The `$USER` variable needs to be replaced with your username (or simply the correct path to your installation).

## Configuration

The configuration is stored in a YAML file. A default is given in [`configs/default_config.yaml`](configs/default_config.yaml).

Notable configuration sections:
* `URLs`: URLs of all external services used by the system, mostly from [Lindat](http://lindat.mff.cuni.cz/).
    * `SOLR` -- this is the URL of the Solr QA system, now configured to UFAL, reconfigure for your setup to make sure your updated QA data are reflected
* `QA`: Configuration of the QA models:
    * `MODEL_TYPE` -- type of the model to extract answers from Wiki search results (defaults to OpenAI, can be changed to Huggingface or a locally trained model). If this is `null`, no QA is applied and the best reranked Wiki result is returned as a whole. Specific models are then configured using `OPENAI_MODEL`, `HUGGINGFACE_MODEL` or `LOCAL_MODEL`.
    * `SITE_TO_PREF` -- this will set preferences for QA data sources (e.g. first `NPI`, then `WIKI`).
    * `INAPPROPRIATE_KWS` -- list of regexes used to filter QA results; if any of these words is included, the result will be ruled out
    * `DISTANCE_THRESHOLD` -- a cosine distance threshold for reranking -- any Solr result beyond the threshold will be ruled out; setting this to `0` will effectively ignore out-of-domain results, so e.g. Wiki search reuslts will never be returned if the intent is `qa_npi`.
    * `DISTANCE_THRESHOLD_INDOMAN` -- same thing, but applied if intent matches the QA source (e.g. intent is `qa_npi` and the source is `NPI`). Setting this to `>1` will make the system use all results from the given source.
    * `SENTENCE_REPR_MODEL` -- reranking model (for Wiki results, used globally if no in-domain model is set)
    * `SENTENCE_REPR_MODEL` -- reranking model for in-domain results (e.g. `NPI`) 
* `HC_RESPONSES_PATH`: Set of handcrafted responses for specified intents
* `INTENT_MODEL`:
    * `PATH` -- local path to the trained model
    * `URL` -- URL for download of a trained model from a remote server (currently set to UFAL server)
* `CHITCHAT`: 
    * `MODEL` -- chitchat model settings -- either use a Huggingface model ID, or `"AIML"`.
    * `DECODE_PARAMS` -- Huggingface model decoder parameters
    * `INAPPROPRIATE_KWS` -- if the chitchat reply contains any of these phrases, it'll be ruled out and the model wil lprovide a fallback.
* `CHITCHAT_INTENTS`: list of intents for which the input is passed on to the chitchat model (instead of QA or handcrafted replies)
* `ABBREV_REPLACE`: list of abbreviation replacements for in-domain IR, used in IR data creation as well as on the fly before querying Solr. E.g., `ZŠ` is expanded to `základní škola` both in the data and in the user query, so it doesn't matter if the user uses a full word or the abbreviation.

## Updating the system

To newly prepare the intent classifier training data, edit the [`data/create_merged_intents.py`](data/create_merged_intents.py) 
script to update file locations, and run the script. The script assumes the EDU-AI data Excel file as input
and includes [`data/intents_to_add.tsv`](data/intents_to_add.tsv) to the mix.


To then retrain the intent classifier, run the following command (default parameters, change optionally):
```
python -m edubot.educlf.train --data_fn data/merged_intents.tsv --model robeczech --out_dir $(pwd)/trained_classifierXY
```
Make sure you update the intent model location in the configuration file accordingly.


To update the QA data, you need to run this:
```
python data/create_ir_data.py --config configs/default_config.yaml --custom data/Dotazy\ Wiki\ chatbot.xlsx --wiki data/cs_wiki.tsv data/ir_data.tsv
```
This expects the EDU-AI Excel file (`--custom`), same as the intent classifier, 
as well as a wikipedia dump from [Wikiextractor](https://github.com/attardi/wikiextractor) 
(`--wiki`), postprocessed by [data/wiki_paragraph_extractor.py](data/wiki_paragraph_extractor.py). 

Use the following setting for the postprocessing:
```
./data/wiki_paragraph_extractor.py --input <dump_dir> --output cs_wiki.tsv --bucket_size 1000000000 --buckets 1 --min_len 0 --max_len 1000000 --skip_beg_chars 0
```

You may want to check the abbreviation expansion in the configuration file before you run this.

## Scripts

### Querying

You can use these to query the system:
```
./scripts/ask.sh "user query" 1 8202
python scripts/interactive.py --url 'http://localhost:8202 <--exact|--no-exact>'
```

`ask.sh` is only used for single queries (the parameters specify the user query, the "exact" parameter controlling the use of QA models, and the target port). Localhost is assumed for this. `interactive.py` is used to maintain a continuous conversation. The parameters are essentially the same, but it can query remote servers. Use `Ctrl-C` to stop the conversation.


### Evaluating reranking

This will test the NPI and Wiki queries + their correct reranking:
```
python scripts/test_npi.py data/test_npi.json
python scripts/test_wiki.py data/test_wiki.json
```
Both scripts assume that the system is running on `localhost:8202` -- they need to be changed if the system is running elsewhere.

### Evaluating the QA model (based on configuration)

This will evaluate the full wiki QA pipeline, including Solr, reranking and the QA model (to extract answer from Solr wiki paragraph):
```shell
python -m scripts.evaluate --eval_file data/edubot-data-augmented.json --data_root data/ --data_type edubot --config configs/default_config.yaml
```

## LICENSE & ACKNOWLEDGEMENTS

TODO

