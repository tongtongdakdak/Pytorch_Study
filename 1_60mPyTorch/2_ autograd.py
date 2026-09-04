import torch
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights = ResNet18_Weights.DEFAULT)
data = torch.rand(1,3,64,64) # 배치크기, 채널수, 세로픽셀수, 가로픽셀수 (임의의 텐서)
labels = torch.rand(1,1000) # 배치크기, 클래스 개수(ResNet의 클래스는 1000개)

prediction = model(data)
loss = (prediction - labels).sum() #임의의 Loss
loss.backward()

optim = torch.optim.SGD(model.parameters(), lr= 1e-2, momentum=0.9)
optim.step()

# Autograd에서의 diffrentiation
a = torch.tensor([2.,3.], requires_grad=True)
b = torch.tensor([6.,4.], requires_grad=True)
Q = 3*a**3 - b**2

external_grad = torch.tensor([1.,1.])
Q.backward(gradient=external_grad)

print(9*a**2 == a.grad)
print(-2*b == b.grad)

x = torch.rand(5,5)
y = torch.rand(5,5)
z = torch.rand(5,5, requires_grad=True)

a = x + y
b = x + z
print(f"Does a require grad: {a.requires_grad}")
print(f"Does b require grad: {b.requires_grad}")

from torch import nn, optim

model = resnet18(weights = ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False # freezing

model.fc = nn.Linear(512,10) # 입력크기, 출력크기 
optimizer = optim.SGD(model.parameters(), lr = 1e-2, momentum=0.2)
