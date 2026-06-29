import torch

from esa import TinyESALanguageModel
from esa.utils import count_parameters


# Tiny smoke-test language model example.
text = "entangled state attention " * 20
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

model = TinyESALanguageModel(
    vocab_size=len(chars),
    block_size=8,
    dim=16,
    n_layer=1,
    dropout=0.0,
)

print("parameters:", count_parameters(model))

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

ix = torch.randint(0, len(data) - 9, (2,))
x = torch.stack([data[i : i + 8] for i in ix])
y = torch.stack([data[i + 1 : i + 9] for i in ix])
logits, loss = model(x, y)
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
print(f"loss: {loss.item():.4f}")

prompt = torch.tensor([[stoi["e"]]], dtype=torch.long)
out = model.generate(prompt, max_new_tokens=10, temperature=0.9, top_k=8)[0].tolist()
print("sample:", "".join(itos[i] for i in out))
