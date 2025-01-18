import torch
import os 
import sys 
import matplotlib.pyplot as plt 
# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.to(device)
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


def calc_batch_loss(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    model.to(device)

    logits = model(input_batch).logits[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss 


def calc_loader_loss(data_loader, model, device, num_batches=None):
    model.to(device)
    total_loss = 0

    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(len(data_loader), num_batches)

    for i, (input, target) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_batch_loss(input, target, model, device)
            total_loss += loss 
        else:
            break 
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loader_loss(train_loader, model, device, num_batches=eval_iter) 
        val_loss = calc_loader_loss(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss 


def train_classifier_simple(model, train_loader, val_loader, optimizer, 
                            device, num_epochs, eval_freq, eval_iter):
    
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input, target in train_loader:
            optimizer.zero_grad()
            loss = calc_batch_loss(input, target, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input.shape[0]
            global_step += 1 

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(f"Ep {epoch + 1} (Step {global_step:06d}) : "
                  f"Train Loss {train_loss:.3f}, "
                  f"Val loss {val_loss:.3f}")
        
            train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
            val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)

            print(f"Training Accuracy: {train_accuracy*100:.2f}%  | ", end="")
            print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
            train_accs.append(train_accuracy)
            val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples Seen")

    fig.tight_layout()
    plt.show()