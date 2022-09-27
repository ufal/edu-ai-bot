#!/bin/bash

curl -s -H "Content-Type: application/json" -X POST -d "{\"q\": \"$1\", \"exact\": $2}" 'http://localhost:8202' | python -c 'import json,sys; data = json.load(sys.stdin); json.dump(data, sys.stdout, indent=4, ensure_ascii=False); print("");'
