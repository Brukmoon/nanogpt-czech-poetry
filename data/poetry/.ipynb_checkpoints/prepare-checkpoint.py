import json
import os
import glob
import tiktoken
import numpy as np


def extract_poems(folder_path):
    poem_lines = []
    for file_path in glob.glob(os.path.join(folder_path, '*.json')):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for body in data:
                for stanza in body['body']:
                    for line in stanza:
                        poem_lines.append(line['text'])
    return poem_lines

folder_path = r'C:\Users\Michal\nanoGPT\data\poetry\ccv'  # Replace with your folder path
data = extract_poems(folder_path)
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]
train_data = '\n'.join(train_data)
val_data = '\n'.join(val_data)
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))