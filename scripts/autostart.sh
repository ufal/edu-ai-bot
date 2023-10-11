#!/bin/bash

USER=`whoami`
PORT=8202
API_KEY=$1

if curl "http://localhost:$PORT/ping" 2>/dev/null; then
    echo '.'
else
    echo 'Restarting...'
    cd /home/$USER/edu-ai-bot
    nohup bash -lc "conda activate edubot; OPENAI_API_KEY=$API_KEY python api.py --port $PORT --config configs/default_config.yaml --logfile log.jsonl 2>&1 | tee -a console-log.txt" &
fi


