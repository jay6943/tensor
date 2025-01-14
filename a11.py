import cfg
import a10
import torch

filename = cfg.workspace + 'a10.pt'

model = a10.NeuralNetwork()
model.load_state_dict(torch.load(filename, weights_only=True))
model.eval()

x_test = torch.Tensor([-3.1, 3.0, 1.2, -2.5]).view(4, 1)
print(model(x_test))
