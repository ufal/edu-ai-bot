
## Preparing data for intent classification

- First, get `Dotazy Wiki chatbot.xlsx` off of Google Sheets. 
* Then, run:
```
python create_merged_intents.py
```

## Preparing data for Solr retrieval

- Get `Dotazy Wiki chatbot.xlsx` off of Google Sheets. 
* Get a wikipedia dump into `cs_wiki_paragraphs_redirects_resolved.tsv`
* Run this:
```
python create_ir_data.py
```
