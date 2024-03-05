import matplotlib.pyplot as plt
import torch
import numpy as np
import ipywidgets as widgets
from IPython.display import clear_output

def visu_inference(infer, model, x):
    temp = 0.8
    top_k = 10
    temp_slider = widgets.FloatSlider(
        value=temp,
        min = 0.01,
        max=100,
        description='Temperature:')
    topk_slider = widgets.IntSlider(
        value=top_k,
        min = 1,
        max=100,
        description='Top K:')
    out = widgets.Output()
    def on_change_temperature(v):
        with out:
            global temp
            clear_output()
            temp = v.new
            infer(model, x, temp, top_k)
    def on_change_topk(v):
        with out:
            global top_k
            clear_output()
            top_k = v.new
            infer(model, x, temp, top_k)
    temp_slider.observe(on_change_temperature, names='value')
    topk_slider.observe(on_change_topk, names='value')
    
    display(temp_slider, topk_slider, out)

def visu_tok_embedding(te):
    # Visualize token embeddin
    tnp = te.cpu().detach().numpy().copy()
    tnp.resize((te.shape[1],te.shape[2]))
    fig, ax = plt.subplots(figsize=(30, 30))
    ax.title.set_text('Token embedding')
    im = ax.imshow(tnp)
    ax.set_xlabel('n_emb')
    ax.set_ylabel('T')
    fig.colorbar(im, fraction=0.0046, pad=0.04)

def visu_lpos_embedding(pe):
    # Visualize position embedding
    pnp = pe.cpu().detach().numpy().copy()
    pnp.resize((pe.shape[0],pe.shape[1]))
    fig, ax = plt.subplots(figsize=(30, 30))
    ax.title.set_text('Position embedding (learned)')
    im = ax.imshow(pnp)
    ax.set_xlabel('n_emb')
    ax.set_ylabel('T')
    fig.colorbar(im, fraction=0.0046, pad=0.04)

def visu_fpos_embedding(seq_len, n_embd, getPositionEmbedding, n=100):
    P = getPositionEmbedding(seq_len=seq_len, d=n_embd, n=n)
    fig, ax = plt.subplots(figsize=(30, 30))
    ax.title.set_text('Position embedding (sin/cos)')
    im = ax.imshow(P)
    ax.set_xlabel('n_emb')
    ax.set_ylabel('T')
    fig.colorbar(im, fraction=0.0046, pad=0.04)

def visu_lres(y):
    ynp = y.cpu().detach().numpy().copy()
    ynp.resize((y.shape[1],y.shape[2]))
    fig, ax = plt.subplots(figsize=(30, 30))
    ax.title.set_text('Result (learned)')
    im = ax.imshow(ynp)
    ax.set_xlabel('n_emb')
    ax.set_ylabel('T')
    fig.colorbar(im, fraction=0.0046, pad=0.04)

def visu_fres(te, getPositionEmbedding, n=100):
    P = getPositionEmbedding(seq_len=te.shape[1], d=te.shape[2], n=n)
    PC = torch.Tensor(P).to(device='cuda')
    y2 = PC + te
    ynp = y2.cpu().detach().numpy().copy()
    ynp.resize((te.shape[1],te.shape[2]))
    fig, ax = plt.subplots(figsize=(30, 30))
    ax.title.set_text('Result (sin/cos)')
    im = ax.imshow(ynp)
    ax.set_xlabel('n_emb')
    ax.set_ylabel('T')
    fig.colorbar(im, fraction=0.0046, pad=0.04)

def visualise_probs(probs, stoi):
    pnp = probs.cpu().detach().numpy()[0]
    plt.bar(np.array(list(stoi.keys()))[np.nonzero(pnp)],pnp[np.nonzero(pnp)])
    plt.show()

def visu_all(te, pe, getPositionEmbedding, y, n=100):
    visu_tok_embedding(te)
    visu_lpos_embedding(pe)
    visu_fpos_embedding(pe.shape[0], pe.shape[1], getPositionEmbedding, n=100)
    visu_lres(y)
    visu_fres(te, getPositionEmbedding)