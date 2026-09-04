# Neural Networks
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # 입력 이미지 채널 1개, 출력 채널 6개, 5*5의 정사각형 행렬
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # 10개의 클래스
    
    # def forward(self, input):
    #     c1 = F.relu(self.conv1(input))
    #     s2 = F.max_pool2d(c1, (2, 2))
    #     c3 = F.relu(self.conv2(s2))
    #     s4 = F.max_pool2d(c3, 2)
    #     s4 = torch.flatten(s4,1)
    #     f5 = F.relu(self.fc1(s4))
    #     f6 = F.relu(self.fc2(f5))
    #     output = self.fc3(f6)
    #     return output
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())

input = torch.randn(1,1,32,32) # 배치 크기 채널수 가로세로
out = net(input)
print(f"out: {out}")

net.zero_grad()
out.backward(torch.randn(1,10))

output = net(input)
target = torch.randn(10)
target = target.view(1,-1)
criterion = nn.MSELoss()

loss = criterion(output, target)
print(f"Loss: {loss}")

print(loss.grad_fn) # 
print(loss.grad_fn.next_functions[0][0])
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])
# 첫번째 [0]: 연산된 텐서중 인덱스번째 텐서
# 두번째 [0]: 부모 연산의 인데스번째 출력인지 나타내는 텐서

net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# weight = weight - learning_rate * gradient
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)



import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.01) # optimizer 생성
optimizer.zero_grad() # 변화도 버퍼를 0으로
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step() # 업데이트 진행