from black import out
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.nn.functional as F
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5))
])


class Net(nn.Module):
    def  __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim * input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

train_data = datasets.FashionMNIST('./data',train=True, transform=transform,
                                   download=True)
test_data = datasets.FashionMNIST('./data',train=False, transform=transform,
                                   download=True)

batch_size = 64
lr = 1e-3
n_class = 10
input_dim = 28
output_dim = 10
epochs = 20

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

device='cuda' if torch.cuda.is_available() else 'cpu'
net = Net(input_dim, output_dim).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)


criterion = nn.CrossEntropyLoss()
loss_all = []
net.train()
for epoch in range(epochs):
    loss_epoch = 0
    for (images, labels) in train_loader:
        images, labels = images.to(device), labels.to(device)
        images = images.reshape(-1, input_dim*input_dim)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.detach().item()
    print(f"epoch = {epoch}, loss = {loss:.4f}")
    
net.eval()
accuracy = 0
with torch.no_grad():
    for (images, labels) in test_loader:
        images, labels = images.to(device), labels.to(device)
        images = images.reshape(-1, input_dim*input_dim)
        outputs = net(images)
        _, predict = torch.max(outputs, dim=1)
        
        accuracy += torch.sum(predict==labels).float()/batch_size
        
accuracy = accuracy / len(test_loader)        
print(accuracy)

    

