#!/bin/bash

curl -s -H "Content-Type: application/json" -X POST "http://localhost:${3}/" -d "{\"q\": \"$1\", \"exact\": $2}"  | python -c 'import json,sys; data = json.load(sys.stdin); json.dump(data, sys.stdout, indent=4, ensure_ascii=False); print("");'
