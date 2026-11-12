import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText

from model import Model

MODEL = 'Qwen/Qwen3.5-4B'


class Qwen(Model):
    """Zero-shot Qwen3.5-4B: show the cartoon and the captions, ask for one letter. Ignores train."""

    def __init__(self, train):
        super().__init__(train)
        self.proc = AutoProcessor.from_pretrained(MODEL, min_pixels=256 * 32 * 32, max_pixels=512 * 512)
        self.model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map='cuda')

    def predict(self, test):
        return [self.ask(ex) for ex in tqdm(test)]

    def ask(self, ex):
        if 'choices' in ex:
            q, opts, labels = 'Choose the caption originally written for this cartoon.', ex['choices'], 'A, B, C, D, or E'
        else:
            q, opts, labels = ('Choose the caption that New Yorker caption-contest readers rated as funnier for this cartoon.',
                               [ex['caption_A'], ex['caption_B']], 'A or B')
        letters = 'ABCDE'[:len(opts)]
        text = q + '\n' + '\n'.join(f'{l}: {c}' for l, c in zip(letters, opts)) + f'\nReply with exactly one letter ({labels}).'
        img = Image.open(ex['image']).convert('RGB')
        msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': img}, {'type': 'text', 'text': text}]}]
        inp = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                            return_tensors='pt', enable_thinking=False).to(self.model.device)
        out = self.model.generate(**inp, max_new_tokens=16, do_sample=False)
        ans = self.proc.decode(out[0, inp['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        return ans if len(ans) == 1 and ans in letters else None  # anything else counts as wrong
