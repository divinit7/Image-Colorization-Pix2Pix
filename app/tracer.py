import sys

import torch

sys.path.append('C:\\Users\\chauh\\Documents\\Github\\tutorial_notebooks\\Colorizer\\src')
from train import *

net_G = build_res_unet(n_input=1, n_output=2, size=256)
net_G.load_state_dict(torch.load("res18-unet.pt", map_location=device))
model = MainModel(net_G=net_G)
model.load_state_dict(torch.load('final-trained-model.pt', map_location=device))
example_input = torch.rand(1, 3, 224, 224)


model.eval()
traced_model = torch.jit.trace(model,example_inputs= model.setup_input(example_input))
traced_model.save("./traced_model.pt")


