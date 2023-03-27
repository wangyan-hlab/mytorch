#%%
# Importing libs
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import ViTModel

batch_size = 32
num_last_hidden = 768   # 256*3
num_labels = 131
image_size = 224

#%%
# Modifying the pre-trained model into a standard classifier
class ViTForImageClassification(nn.Module):
    def __init__(self, num_labels=num_labels, vector_length=num_last_hidden):
        super(ViTForImageClassification, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.last_layer = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels
    
    def forward(self, x):
        x = self.vit(x).last_hidden_state[:, 0]
        x = self.last_layer(x)
        
        return x
    
# Defining the model
model = ViTForImageClassification()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Loading the saved model
network_state_dict = torch.load('./fruits_model/model_vit_fruits_man-'+str(num_labels)+'.pth')
model.load_state_dict(network_state_dict)

#%%
# Nominal input to the model, only matching the size 
nominal_input = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
nominal_input = nominal_input.to(device)
# Export the model via tracing
torch.onnx.export(model,
                  nominal_input,
                  "fruits_model/vit_fruits_man-"+str(num_labels)+".onnx",
                  export_params=True,
                  opset_version=10,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input' : {0 : 'batch_size'},
                                'output' : {0 : 'batch_size'}}
                  )

#%%
# Check the onnx model
import onnx

onnx_model = onnx.load("fruits_model/vit_fruits_man-"+str(num_labels)+".onnx")
onnx.checker.check_model(onnx_model)

#%%
# Getting data
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_dataset = torchvision.datasets.ImageFolder('../../data/fruits/fruits-360_dataset/fruits-360/Test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
example_data, example_targets = example_data.to(device), example_targets.to(device)

#%%
# Run the onnx model
import onnxruntime

ort_session = onnxruntime.InferenceSession("fruits_model/vit_fruits_man-"+str(num_labels)+".onnx",
                                        providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
                                           )
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(example_data)}
ort_outs = ort_session.run(None, ort_inputs)
onnx_out = ort_outs[0]           # onnx model output
_, onnx_pred = torch.max(torch.from_numpy(onnx_out), 1)

#%%
# Compare ONNX Runtime and PyTorch results
model.eval()
with torch.no_grad():
    torch_out = model(example_data) # torch model output
    _, torch_pred = torch.max(torch_out, 1)
    
np.testing.assert_allclose(to_numpy(torch_pred), to_numpy(onnx_pred), rtol=1e-03, atol=1e-05)
print("Exported model has been tested with ONNXRuntime, and the result looks good!")

#%%
# Visualizing the prediction result of the onnx model
example_data = example_data.detach().cpu()
import matplotlib.pyplot as plt
fig = plt.figure()
# Draw the first 4 of a batch
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.tight_layout()
    plt.imshow(np.transpose(example_data[i].numpy(), [1,2,0]), interpolation='none')
    plt.title("Prediction: {}\nGroundtruth: {}".format(onnx_pred[i], example_targets[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()
# %%
