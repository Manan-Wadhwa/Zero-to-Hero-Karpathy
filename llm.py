# %%
with open ('input.txt', 'r') as f:
    text = f.read()
# %%
text[:1000]

# %%
len(text)

# %%
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars), vocab_size)
# %%

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
def encode(s):
    out = []
    for c in s:
        out.append(stoi[c])
    return out
print(encode("hello world"))
def decode(l):
    
    out = ""
    for i in l:
        out += itos[int(i)]
    return out
print(decode(encode("hello world")))

# %%
import torch 
data = torch.tensor(encode(text))
print(data.shape, data.dtype)
print(data[:1000])  
# %%
split = int(0.9*len(data))
train_data = data[:split]
val_data = data[split:] 


# %%
block_size = 8
train_data[:block_size+1]

# %%
x= train_data[:block_size]
y= train_data[1:block_size+1]
for i in range(block_size):
    context = x[:i+1]
    target = y[i]
    print(context , target)
# %%
torch.manual_seed(1337)
batch_size = 4
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y  

xb,yb = get_batch('train')
print('inputs:')
print(xb)
print('targets:')
print(yb)
# %%
