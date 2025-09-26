from model import GPT, Config
import torch
import tiktoken
from tqdm import tqdm

class DataPreprocessor:
    def __init__(self, file_path, B, T, train_ratio=0.9):
        self.B = B
        self.T = T
        self.enc = tiktoken.get_encoding("gpt2")
        self.tokens = self.encode_text(self.load_data(file_path))

        n = len(self.tokens)
        split = int(train_ratio * n)
        self.train_tokens = self.tokens[:split]
        self.val_tokens = self.tokens[split:]

        self.current_position = 0
        print(f"Loaded {n} tokens")
        print(f"1 epoch = {n // (B*T)} batches")

    def load_data(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def encode_text(self, text):
        return torch.tensor(self.enc.encode(text), dtype=torch.long)

    def next_batch(self, split="train"):
        data = self.train_tokens if split == "train" else self.val_tokens
        start = self.current_position
        end = start + self.B * self.T + 1
        if end >= len(data):
            self.current_position = 0
            start, end = 0, self.B * self.T + 1
        buf = data[start:end]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        self.current_position += self.B * self.T
        return x, y

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = DataPreprocessor("datasets/moby_dick.txt", B=4, T=32)
model = GPT(Config()).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
num_epochs = 5
batches_per_epoch = len(data.train_tokens) // (data.B * data.T)

for epoch in range(1, num_epochs + 1):
    epoch_loss = 0.0

    # tqdm progress bar for each epoch
    progress_bar = tqdm(range(batches_per_epoch), desc=f"Epoch {epoch}/{num_epochs}", leave=True)

    for step in progress_bar:
        x, y = data.next_batch(split="train")
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad() #Zeros the gradiants, really important
        logits, loss = model(x, y)
        loss.backward() #Adds to the gradiants that were already zeroed. which is why zeroing is important.
        optimizer.step() #Updates the parameters.
        # accumulate loss for reporting
        epoch_loss += loss.item()
        # update progress bar with current batch loss
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    # average epoch loss
    avg_loss = epoch_loss / batches_per_epoch
    print(f"Epoch {epoch} finished | Avg loss: {avg_loss:.4f}")
