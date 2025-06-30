import torch


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        
        self.layers = torch.nn.Sequential(
            # first hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            
            # second hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            
            # output layer
            torch.nn.Linear(20, num_outputs)
        )
    
    def forward(self, x):
        logits = self.layers(x)
        return logits


def main():
    model = NeuralNetwork(50, 3)
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of trainable model parameters:", num_params)
    # print(model.layers[0].weight)
    # print(model.layers[0].weight.shape)
    # print(model.layers[0].bias)
    

if __name__=="__main__":
    main()