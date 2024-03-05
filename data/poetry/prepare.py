#! /bin/python3
import json
import os
import glob
import tiktoken
import numpy as np


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

from collections import Counter, defaultdict

def get_vocab(text):
    # Initialize vocabulary with frequency of each word in text
    vocab = Counter(text.split())
    return {word: freq for word, freq in vocab.items()}

def get_stats(vocab):
    # Get frequency of adjacent symbol pairs (bigrams) in vocabulary
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, vocab):
    # Merge most frequent pair in all vocabulary words and update frequency
    new_vocab = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word in vocab:
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = vocab[word]
    return new_vocab


folder_path = r'./corpusCzechVerse/ccv'  # Replace with your folder path
data = extract_poems(folder_path)
n = len(data)
text = '\n'.join(data)
train_data = text[:int(n*0.9)]
val_data = text[int(n*0.9):]
#train_data = '\n'.join(train_data)
#val_data = '\n'.join(val_data)

# Sample text data
#text = train_data.join(val_data)

# Convert each word in initial vocabulary to space-separated string of characters
vocab = get_vocab(text)
vocab = {' '.join(word): freq for word, freq in vocab.items()}
print("Initial vocabulary:", vocab)

# Number of BPE iterations
num_merges = 10

for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    # Get the most frequent pair
    best_pair = max(pairs, key=pairs.get)
    vocab = merge_vocab(best_pair, vocab)
    print(f"After iteration {i+1}, Best pair: {best_pair}")
    print("Updated vocabulary:", vocab)

"""enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))"""
