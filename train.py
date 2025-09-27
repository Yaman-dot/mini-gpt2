from model import GPT, Config
import math
import torch
import tiktoken
from tqdm import tqdm
import time
class DataManager:
    def __init__(self, file_path, B, T, total_batch_size, train_ratio=0.9):
        self.B = B
        self.T = T
        self.enc = tiktoken.get_encoding("gpt2")
        self.tokens = self.encode_text(self.load_data(file_path))
        
        self.total_batch_size = total_batch_size  # Desired total batch size in tokens
        assert total_batch_size % (B * T) == 0, "Total batch size must be divisible by B * T"
        self.grad_accum_step = total_batch_size // (B * T)  # Number of gradient 

        n = len(self.tokens)
        split = int(train_ratio * n)
        self.train_tokens = self.tokens[:split]
        self.val_tokens = self.tokens[split:]

        #self.current_position = 0
        self.train_position = 0
        self.val_position = 0
        print(f"Loaded {n} tokens")
        print(f"Total desired batch size: {total_batch_size}")
        print(f"Calculated gradient accumulation steps: {self.grad_accum_step}")
        print(f"1 epoch = {n // (B * T)} batches")

    def load_data(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()[:10000]
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: File {file_path} not found.")
        except Exception as e:
            raise Exception(f"Error reading file {file_path}: {str(e)}")
    def encode_text(self, text):
        return torch.tensor(self.enc.encode(text), dtype=torch.long)

    def _get_batch(self, data, position_ref):
        data_len = len(data)
        if position_ref[0] + self.B * self.T + 1 > data_len:
            position_ref[0] = 0
        
        start = position_ref[0]
        end = start + self.B * self.T + 1
        
        buf = data[start:end]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        
        position_ref[0] += self.B * self.T
        return x, y

    def next_train_batch(self):
        return self._get_batch(self.train_tokens, [self.train_position])

    def next_val_batch(self):
        return self._get_batch(self.val_tokens, [self.val_position])
    @staticmethod
    def get_lr(it, warmup_steps=10, max_lr=6e-4, min_lr=6e-5, max_steps=50):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > max_steps:
            return min_lr
        # 3) in between, use cosine decay down to min_lr
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (max_lr - min_lr)
    def train(self, optimizer, model, num_epochs):
        batches_per_epoch = len(data.train_tokens) // (data.B * data.T)
        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0.0
            # tqdm progress bar for each epoch
            progress_bar = tqdm(range(batches_per_epoch), desc=f"Epoch {epoch}/{num_epochs}", leave=True)

            epoch_start_time = time.time()  # Start timing for epoch
            for step in progress_bar:
                batch_start_time = time.time()
                optimizer.zero_grad() #Zeros the gradiants, really important
                micro_batch_loss = 0.0
                for micro_step in range(self.grad_accum_step):
                    x, y = self.next_train_batch()
                    if torch.cuda.is_available():
                        x, y = x.to(device), y.to(device)

                    if torch.cuda.is_available():
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            logits, loss = model(x, y)
                    else:
                        logits, loss = model(x, y)
                    loss = loss / self.grad_accum_step
                    micro_batch_loss += loss.detach()
                    loss.backward() #Adds to the gradiants that were already zeroed. which is why zeroing is important.
           
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #Clip the gradiants to avoid exploding gradiants.

                lr = self.get_lr(step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                optimizer.step() #Updates the parameters.

                torch.cuda.synchronize() if torch.cuda.is_available() else None #wait for the gpu to finish its work
                batch_time = (time.time() - batch_start_time) * 1000
                avg_micro_batch_loss = micro_batch_loss / self.grad_accum_step
                # accumulate loss for reporting
                epoch_loss += loss.items()
                # update progress bar with current batch loss
                progress_bar.set_postfix({"loss": f"{micro_batch_loss.item():.4f}", "norm": f"{norm:.4f}", "lr": f"{lr:.4f}", "ms": f"{batch_time:.2f}"})

            epoch_time = (time.time() - epoch_start_time) * 1000  # Convert to ms
            # average epoch loss
            avg_loss = epoch_loss / batches_per_epoch
            #print(f"Epoch {epoch} finished | Avg loss: {avg_loss:.4f}")


            # Validation loop
            model.eval()
            val_loss = 0.0
            val_batches = max(1, len(self.val_tokens) // (self.B * self.T))
            with torch.no_grad():
                for _ in range(val_batches):
                    x, y = self.next_val_batch()
                    if torch.cuda.is_available():
                        x, y = x.to(device), y.to(device)
                    _, loss = model(x, y)
                    val_loss += loss.item()
            avg_val_loss = val_loss / val_batches
            print(f"Epoch {epoch} completed in {epoch_time:.2f}ms | Avg train loss: {avg_loss:.4f} | Avg val loss: {avg_val_loss:.4f}")


# Main script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

torch.set_float32_matmul_precision("high")
data = DataManager("datasets/moby_dick.txt", 4, 32, 524288, 0.9)
model = GPT(Config(vocab_size=50304)).to(device)
#model = torch.compile(model)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
num_epochs = 1
data.train(optimizer, model, num_epochs)
