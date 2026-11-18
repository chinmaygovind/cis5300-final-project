import base64
import os
import time

import requests

# Azure deployment of gpt-4.1-nano; endpoint + key live outside the repo
DEPLOYMENT = 'CIS-6770-Chinmay-nano'
env = dict(l.strip().split('=', 1) for l in open(os.path.expanduser('~/.azure-cis6770-env')) if '=' in l)
URL = env['AZURE_OPENAI_ENDPOINT'].rstrip('/') + '/openai/v1/chat/completions'
HEAD = {'api-key': env['AZURE_OPENAI_API_KEY']}


class GPTNano:
    """Zero-shot gpt-4.1-nano with the same prompt as Qwen. Ignores train.
    Image sent at detail='low' (512x512), about the same resolution Qwen sees."""

    workers = 8  # parallel requests; retries below absorb Azure's 429s

    def __init__(self, train, task):
        self.task = task

    def predict_one(self, ex):
        if self.task == 'matching':
            q, opts, labels = 'Choose the caption originally written for this cartoon.', ex['choices'], 'A, B, C, D, or E'
        else:
            q, opts, labels = ('Choose the caption that New Yorker caption-contest readers rated as funnier for this cartoon.',
                               [ex['caption_A'], ex['caption_B']], 'A or B')
        letters = 'ABCDE'[:len(opts)]
        text = q + '\n' + '\n'.join(f'{l}: {c}' for l, c in zip(letters, opts)) + f'\nReply with exactly one letter ({labels}).'
        img = base64.b64encode(open(ex['image'], 'rb').read()).decode()
        body = {'model': DEPLOYMENT, 'temperature': 0, 'max_tokens': 4, 'messages': [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{img}', 'detail': 'low'}},
            {'type': 'text', 'text': text}]}]}
        for _ in range(30):
            try:
                r = requests.post(URL, headers=HEAD, json=body, timeout=120)
            except requests.RequestException:
                time.sleep(15)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(float(r.headers.get('retry-after', 10)) + 1)
                continue
            if r.status_code == 400 and ('content_filter' in r.text or 'content_policy_violation' in r.text):
                return None  # Azure's safety filter refused the image or prompt (e.g. cartoon 617): counts as wrong
            r.raise_for_status()
            ans = (r.json()['choices'][0]['message'].get('content') or '').strip()
            return ans if len(ans) == 1 and ans in letters else None  # anything else counts as wrong
        raise RuntimeError('azure kept rate limiting')
