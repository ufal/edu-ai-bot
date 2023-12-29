#!/bin/bash

USER=`whoami`
EMAIL="user@domain.cz"
PORT=8202
API_KEY=$1
SOLR_URL="http://quest.ms.mff.cuni.cz/namuddis/qasolr/wiki_test/query"

date

if curl "$SOLR_URL" 2>/dev/null; then
    echo '.'
else
    echo 'Solr not running, sending alert...'
    echo -e "Subject: EDU-AI Solr not running\nCheck the server" | /usr/sbin/sendmail "$EMAIL"
fi

if curl "http://localhost:$PORT/ping" 2>/dev/null; then
    echo '.'
else
    echo 'Restarting...'
    echo -e "Subject: Restarting EDU-AI bot\nYou may want to check the server" | /usr/sbin/sendmail "$EMAIL"
    cd /home/$USER/edu-ai-bot
    nohup bash -lc "conda activate edubot; OPENAI_API_KEY=$API_KEY python api.py --port $PORT --config configs/default_config.yaml --logfile log.jsonl 2>&1 | tee -a console-log.txt" &
fi


