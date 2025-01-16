import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Generate text
input_text = "Every effort move you"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_ids)

# Decode the output
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded_output)

for param in model.parameters():
    param.requires_grad = False

torch.manual_seed(123)
num_classes = 2
model.lm_head = torch.nn.Linear(
    in_features=768,
    out_features=num_classes
)

for param in model.transformer.h[-1].parameters():
    param.requires_grad = True

for param in model.transformer.ln_f.parameters():
    param.requires_grad = True

inputs = tokenizer.encode("Do you have time")
inputs = torch.tensor(inputs).unsqueeze(0)
print("Inputs:", inputs)
print("Inputs dimensions:", inputs.shape)

with torch.no_grad():
    outputs = model(inputs)
print("Outputs:\n", outputs.logits)
print("Outputs dimensions:", outputs.logits.shape)
print("Last output token:", outputs.logits[:, -1, :])


probas = torch.softmax(outputs.logits[:, -1, :], dim=-1)
label = torch.argmax(probas)
print("Class label:", label.item())


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch).logits[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions/num_examples
