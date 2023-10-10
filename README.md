# GPT-ish

Perplexity-based ChatGPT detector! It's a fork of Meta's LLaMA repo that returns inferenced probabilities for each token in a text instead of generating text.

Or as the app describes it:

> When you ask an AI to write something for you, a neural network takes in
> the previous text (if any) and then, for every possible token (think of
> them as parts of a word), generates the probability of that particular
> token coming next. It then picks the most\* probable token and does this
> process again until your whole text is generated.
>
> The idea that the AI will always pick one of the most likely tokens is
> the basis for GPT-ish. By running a modified version of a GPT-like
> text generator, we can get the probability distributions for each token
> in the text and allow us to check if they're all just the most likely
> choices (like what an AI would do) or if there's more human-like
> variation.
>
> \*Actually, a technique called temperature sampling picks some less
> likely tokens at times to make your text more interesting.

You can see a demo at [ben476/gpt-ish-solid](https://github.com/ben476/gpt-ish-solid).

## Usage

In a conda env with pytorch / cuda available, run

```
pip install -r requirements.txt
```

Then in this repository

```
pip install -e .
```

The API is in `web.py` which is based on `example.py` from the original LLaMA repository.

```
python -m torch.distributed.run --nproc_per_node 1 example.py --ckpt_dir /local/scratch/hongbenj/llama/7B --tokenizer_path /local/scratch/hongbenj/llama/tokenizer.model
```

## License

See the [LICENSE](LICENSE) file.
