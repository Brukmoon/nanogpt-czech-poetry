from python import json
from python import os
from python import glob
from python import tiktoken
from python import numpy as np

def extract_poems(folder_path):
    poem_lines = []
    for file_path in glob.glob(os.path.join(folder_path, '*.json')):
        with open(file_path, 'r', encoding='us-ascii') as file:
            data = json.load(file)
            for body in data:
                for stanza in body['body']:
                    for line in stanza:
                        poem_lines.append(line['text'])
    return poem_lines

folder_path = r'./corpusCzechVerse/ccv'  # Replace with your folder path
data = extract_poems(folder_path)
n = len(data)
text = '\n'.join(data)
train_text = '\n'.join(data[:int(n*0.9)])
val_text = '\n'.join(data[int(n*0.9):])
keys = sorted(set(list(text)))
from python import re
from collections import Counter, defaultdict


def build_vocab(corpus: str) -> dict:
    """Step 1. Build vocab from text corpus"""

    # Separate each char in word by space and add mark end of token
    # I have included a character blacklist there, I know it might not be a good idea, but I hated having m.-type tokens.
    tokens = [" ".join(word.replace(".","").replace(",","").replace(";","")) + " </w>" for word in corpus.split()]

    # Count frequency of tokens in corpus
    vocab = Counter(tokens)

    return vocab


def get_stats(vocab: dict) -> dict:
    """Step 2. Get counts of pairs of consecutive symbols"""

    pairs = defaultdict(int)
    for word, frequency in vocab.items():
        symbols = word.split()

        # Counting up occurrences of pairs
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += frequency

    return pairs


def merge_vocab(pair, v_in):
    """Step 3. Merge all occurrences of the most frequent pair"""

    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

    for word in v_in:
        # replace most frequent pair in all vocabulary
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]

    return v_out
vocab = build_vocab(text)
num_merges = 500  # Hyperparameter
for i in range(num_merges):
    pairs = get_stats(vocab)  # Step 2
    if not pairs:
        break
    # step 3
    best = max(pairs, key=pairs.get)
    print(f"Iteration {i} out of {num_merges}. Pair = {best}, Frequency = {pairs.get(best)}")
    vocab = merge_vocab(best, vocab)
def extract_unique_tokens(dictionary):
    unique_tokens = set(keys)
    for key in dictionary.keys():
        # Remove "</w>" tokens from the key
        key = key.replace("</w>", "")
        
        # Extract unique characters/bigrams/trigrams separated by a space
        tokens = key.split()
        for token in tokens:
            unique_tokens.add(token)
    return unique_tokens

d = extract_unique_tokens(vocab)
print(d)
