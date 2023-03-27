# Vision Transformer manually trained
# Using the pre-trained model 'google/vit-base-patch16-224'
# Training a standard classifier by placing a linear layer on top of the pre-trained encoder
#%%
# Importing libs
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from transformers import ViTModel

#%%
# Hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 5
num_last_hidden = 768   # 256*3
num_labels = 131
image_size = 224
log_interval = 200
# Getting data
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# Dataset downloaded from https://www.kaggle.com/datasets/moltean/fruits
train_dataset = torchvision.datasets.ImageFolder('../../data/fruits/fruits-360_dataset/fruits-360/Training', transform=transform)
test_dataset = torchvision.datasets.ImageFolder('../../data/fruits/fruits-360_dataset/fruits-360/Test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(len(train_loader.dataset), len(test_loader.dataset))
print(len(train_loader), len(test_loader))
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
    
#%%
# Defining the model
model = ViTForImageClassification()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#%%
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

#%%
# Training
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(num_epochs + 1)]

for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        inputs, labels = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(model.state_dict(), './fruits_model/model_vit_fruits_man-'+str(num_labels)+'.pth')
            torch.save(optimizer.state_dict(), './fruits_model/optimizer_vit_fruits_man-'+str(num_labels)+'.pth')
    print(f'Epoch {epoch + 1} loss: {running_loss / len(train_loader)}')

# %%
# Loading the saved model
network_state_dict = torch.load('./fruits_model/model_vit_fruits_man-'+str(num_labels)+'.pth')
model.load_state_dict(network_state_dict)

#%% 
# Evaluating over the test dataset
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}')
# >>> Accuracy 98%

# %%
# Visualizing the prediction
examples = enumerate(test_loader)
for i in range(1):
    batch_idx, (example_data, example_targets) = next(examples)
    example_data, example_targets = example_data.to(device), example_targets.to(device)
    model.eval()
    with torch.no_grad():
        output = model(example_data)
        print(output)
        _, predicted = torch.max(output, 1)
        print(predicted)
    example_data = example_data.detach().cpu()
    import matplotlib.pyplot as plt
    fig = plt.figure()
    # Draw the first 4 of a batch
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.tight_layout()
        plt.imshow(np.transpose(example_data[i].numpy(), [1,2,0]), interpolation='none')
        plt.title("Prediction: {}\nGroundtruth: {}".format(predicted[i], example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()
# %%
