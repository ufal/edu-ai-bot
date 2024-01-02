# EDU-AI-BOT data

Note: see the [main README](../README.md) for `create_ir_data.py` and `create_merged_intents.py`, which 
create/update the data for QA and for the intent classifier, respectively.

The data in this directory:
* `edubot-data-augmented.json` is used to test the QA models (see `scripts/evaluate.py`)
* `gendered_words.tsv` are used for translation postprocessing in chitchat
* `handcrafted_responses.yml` are used for handling specified intents directly using handcrafted responses (currently date/time responses only)
* `identity_verbs.txt` are used for translation postprocessing in chitchat
* `intents_to_add.tsv` are intent classification training data, used in conjunction with the EDU-AI spreadsheet (not included in the repo)
* `stopwords.txt` are used for filtering QA questions
* `test_npi.json` are test data for testing in-domain reranking for NPI QA
* `test_wikijson` are test data for testing reranking for Wikipedia QA
