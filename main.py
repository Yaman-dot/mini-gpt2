import torch
import tiktoken
from model import GPT
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(model_location = "gpt2"):
    ####default is gpt-2 unless you have trained your own variation you should specify the path in model_location
    return GPT.load_pretrained_model(model_location).to(device)
def reply(prompt, model, max_new_tokens=50):
    #prompt = f"User: {prompt}\AI:"
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)  # (1, seq_len)

    # Generate new tokens
    generated = model.generate(tokens, max_new_tokens=max_new_tokens)  # (1, seq_len + max_new_tokens)

    # Convert to Python list and decode
    generated_tokens = generated[0, tokens.size(1):].tolist()
    decoded = enc.decode(generated_tokens)

    return decoded
if __name__ == "__main__":
    model = load_model("gpt2-medium")
    conversation = ""
    print("Model loaded. Start chatting!")
    while True:
        user_input = input("User>>")
        conversation += user_input + "\n"

        output = reply(conversation, model, max_new_tokens=40)
        print("Model:", output)

        conversation += output + "\n"