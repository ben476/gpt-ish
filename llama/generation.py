# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

import torch

from llama.tokenizer import Tokenizer
from llama.model import Transformer
import time

class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded

    def probs_stream(
        self,
        text: str,
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ):
        params = self.model.params
        print("params", params)

        start = time.time()

        prompt_tokens = [self.tokenizer.encode(text, bos=True, eos=False)]

        window_size = int(9 * params.max_seq_len / 10)

        print("window_size", window_size)
        print("len(prompt_tokens[0])", len(prompt_tokens[0]))

        decoded_tokens = []

        yield self.tokenizer.decode([prompt_tokens[0][0]]), self.tokenizer.decode([prompt_tokens[0][0]]), 1, 1, {}

        for i in range(0, len(prompt_tokens[0]), params.max_seq_len - window_size):
            print("i", i)
            tokens = torch.tensor([prompt_tokens[0][i: i + params.max_seq_len]]).cuda().long()

            print("tokenising took", time.time() - start)

            input_text_mask = tokens != self.tokenizer.pad_id
            start_pos = 1 if i == 0 else window_size
            prev_pos = 0

            print("start_pos", start_pos)
            print("len(prompt_tokens[0])", len(prompt_tokens[0]))

            for cur_pos in range(start_pos, min(params.max_seq_len, len(prompt_tokens[0]) - i)):
                last_tokens = tokens[:, -params.max_seq_len:]
                logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)

                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.argmax(logits, dim=-1)
                
                next_token = next_token.reshape(-1)
                
                sorted_probs, sorted_indices = torch.sort(probs[0], descending=True)

                prev_decode_len = len(self.tokenizer.decode(decoded_tokens))

                decoded_tokens.append(int(tokens[0][cur_pos]))

                decoded = self.tokenizer.decode(decoded_tokens)
                new_decoded = decoded[prev_decode_len:]
                prev_pos = cur_pos

                token = self.tokenizer.decode([int(tokens[0][cur_pos])])
                word = new_decoded
                probability = probs[0][tokens[0][cur_pos]]
                place = torch.where(sorted_indices == int(tokens[0][cur_pos]))[0]
                top5 = {
                    self.tokenizer.decode([int(sorted_indices[i])]): float(sorted_probs[i])
                    for i in range(5)
                }

                yield token, word, float(probability), int(place), top5


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
