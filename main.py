import torch
import tiktoken

from train import GPT

def load_model(model_location = "gpt2"):
    ####default is gpt-2 unless you have trained your own variation you should specify the path in model_location
    return GPT.load_pretrained_model(model_location)
def reply(prompt, model, max_new_tokens=50):
    prompt = f"User: {prompt}\nModel:"
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to('cuda')  # (1, seq_len)
    
    # Generate new tokens
    generated = model.generate(tokens, max_new_tokens=max_new_tokens)  # (1, seq_len + max_new_tokens)
    
    # Convert to Python list and decode
    generated_tokens = generated[0, tokens.size(1):].tolist()
    decoded = enc.decode(generated_tokens)
    
    return decoded
if __name__ == "__main__":
    model = load_model("gpt2")
    model.to('cuda')
    model.eval()
    
    prompt = "Hey your name is GPT2, okay? and my name is Yaman, repeat my name"
    output = reply(prompt, model, max_new_tokens=50)
    print(output)
