import torch
import numpy as np

data = [[1.0,2],[3,4.0]] # 데이터 직접 생성
x_data = torch.tensor(data) # 데이터 자료형 유추
print("직접 생성된 데이터:\n",data,"\n", x_data)

np_arr = np.array(data) # 넘파이 배열로 생성
x_np = torch.from_numpy(np_arr)
print("넘파이로 생성된 데이터:\n",np_arr,"\n", x_np)

x_ones = torch.ones_like(x_data) # shape와 dtype 유지하게 된다
print(f"Ones Tensor:\n{x_ones}")

x_rand = torch.rand_like(x_data, dtype=torch.float16) #dtype을 float 형태로 override하지 않으면 NotIMplementedError이 발생함
print(f"Rand tensor:\n{x_rand}")

''' 
try:
    x_rand = torch.rand_like(x_data, dtype=torch.int)
except NotImplementedError:
    print("테스트용 x_rand 강제 생성")
    x_rand = torch.tensor([1, 2], dtype=torch.int)

if x_rand.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
    raise NotImplementedError("NotImplementedError 발생")

'''

shape=(2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"rand tensor\n{rand_tensor}\n")
print(f"ones tensor\n{ones_tensor}\n")
print(f"zeros tensor\n{zeros_tensor}")

tensor = torch.rand(3,4)

print(f"Shape of tensor : {tensor.shape}")
print(f"Data of tensor : {tensor.dtype}")
print(f"Device tensor on : {tensor.device}")

if torch.cuda.is_available():
    tensor = tensor.to('cuda')
    print(f"tensor is stored in {tensor.device}")
    
tensor = torch.ones(4,4)
tensor.add_(1)
tensor[:,2] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# 텐서곱
print(f"tensor.mul(tensor): \n{tensor.mul(tensor)}\n")
print(f"tensor*tensor: \n{tensor*tensor}")

# 텐서간 행렬곱
print(f"tensor.matmul(tensor.T): \n {tensor.matmul(tensor.T)}")
print(f"tensor @ tensor.T: \n {tensor @ tensor.T}")

# 바꿔치기(inplace 연산)
print(tensor,"\n ")
tensor.add_(5)
print(tensor,"\n")

# 넘파이 변환
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n,1, out=n)
print(f"n: {n}")
print(f"t: {t}")