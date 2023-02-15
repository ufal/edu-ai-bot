#!/usr/bin/env python

import requests
import random
import string
import argparse

def run_conversation(endpoint_url, exact):

    conv_id = 'interactive-' + ''.join(random.choice(string.ascii_lowercase + string.digits) for i in range(20))

    while True:
        query = input('>>> ')
        query = query.strip()
        reply = requests.post(endpoint_url,
                              json={'q': query,
                                    'conversation_id': conv_id,
                                    'exact': exact})
        print(reply.json())



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-u', '--endpoint-url', '--url', type=str, default='http://localhost:8200', help='URL where the server is running')
    ap.add_argument('--exact', action='store_true', help='Require exact results from QA (use QA model, not just solr)')
    ap.add_argument('--no-exact', action='store_false', dest='exact')
    ap.set_defaults(exact=True)
    args = ap.parse_args()

    run_conversation(args.endpoint_url, args.exact)
