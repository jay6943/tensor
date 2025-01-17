import cfg
import torch
import linearR

filename = cfg.works + 'linear.pt'

model = linearR.NeuralNetwork()
model.load_state_dict(torch.load(filename, weights_only=True))
model.eval()

x_test = torch.Tensor([-3.1, 3.0, 1.2, -2.5]).view(4, 1)
print(model(x_test))