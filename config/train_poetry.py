# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-poetry'
eval_interval = 10 # keep frequent because we'll overfit
eval_iters = 50
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'poetry'
wandb_run_name = 'mini-gpt'

dataset = 'poetry'
gradient_accumulation_steps = 1
batch_size = 128
block_size = 128 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 7
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 1500
lr_decay_iters = 1500 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 75 # not super necessary potentially

# on macbook also add
device = 'cuda'  # run on cpu only
#compile = false # do not torch compile the model
