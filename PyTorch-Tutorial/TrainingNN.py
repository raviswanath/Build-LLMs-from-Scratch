import torch
import torch.nn.functional as F
from NeuralNetwork import NeuralNetwork
from TorchDataloader import train_loader, test_loader


def compute_accuracy(model, dataloader):

    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        with torch.no_grad():
            features, labels = features.to(device), labels.to(device)
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    return (correct/total_examples).item() * 100


torch.manual_seed(123)
device = torch.device(
"mps" if torch.backends.mps.is_available() else "cpu"
)
model = NeuralNetwork(num_inputs=2, num_outputs=2)
model = model.to(device)
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.3
)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()

    for batch_idx, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device)
        logits = model(features)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        print(f"Epoch : {epoch+1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Train Loss: {loss:.2f}")

        model.eval()


num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable params : {num_params}")

print(compute_accuracy(model, train_loader))
print(compute_accuracy(model, test_loader))
