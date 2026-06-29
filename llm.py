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
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

# %%
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
# %%
batch_size = 32
for steps in range(10000):
    xb,yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print(loss.item())

# %%
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))
# %%
# consider the following toy example:

torch.manual_seed(1337)
B,T,C = 4,8,2 # batch, time, channels
x = torch.randn(B,T,C)
x.shape

# %%
xbow = torch.zeros((B,T,C))
print(xbow)
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1] # (t,C)
        xbow[b,t] = torch.mean(xprev, 0)
# %%
# version 2: using matrix multiply for a weighted aggregation
wei = torch.tril(torch.ones(T, T))
print(wei)
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x # (B, T, T) @ (B, T, C) ----> (B, T, C)
torch.allclose(xbow, xbow2)
# %%
xbow[0]
# %%
torch.manual_seed(42)
a = torch.tril(torch.ones(3,3))
b = torch.randint(0,10,(3,2)).float()
c = a @ b
print('a=')
print(a)
print('--')
print('b=')
print(b)
print('--')
print('c=')
print(c)
# %%
torch.manual_seed(42)
a = torch.tril(torch.ones(3,3))
a = a / a.sum(1, keepdim=True)
b = torch.randint(0,10,(3,2)).float()
c = a @ b
print('a=')
print(a)
print('--')
print('b=')
print(b)
print('--')
print('c=')
print(c)
#%%
(xbow - xbow2).abs().max() 

# %%
weights = torch.tril(torch.ones(T, T))
weights = weights / weights.sum(1, keepdim=True)
xbow2 = weights @ x
torch.allclose(xbow, xbow2)
torch.allclose(xbow, xbow2, atol=1e-6)
# %%
tril = torch.tril(torch.ones(T, T))
tril
# %%
wei = torch.zeros((T,T))
wei
# %%
wei = wei.masked_fill(tril == 0, float('-inf'))
wei
# %%
wei = F.softmax(wei, dim=-1)
wei
# %%
xbow3 = wei @ x
torch.allclose(xbow, xbow3, atol=1e-6)
# %%
