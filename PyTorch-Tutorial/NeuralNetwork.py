import torch 

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(

            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            # first activation
            torch.nn.ReLU(),

            # 2nd layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(), 

            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits 
    