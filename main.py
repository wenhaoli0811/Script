import torch

# 创建两个 2×2 的张量，并开启自动求导
A = torch.randn(2, 2, requires_grad=True)
B = torch.randn(2, 2, requires_grad=True)

print("Tensor A:")
print(A)

print("\nTensor B:")
print(B)
# 梯度清零
# 防止多次反向传播时梯度累加
if A.grad is not None:
    A.grad.zero_()

if B.grad is not None:
    B.grad.zero_()

# 矩阵乘法
C = torch.matmul(A, B)

print("\n矩阵乘法结果 C:")
print(C)

# 定义损失函数
# 这里将矩阵所有元素求和-
loss = C.sum()

print("\nLoss:")
print(loss)

# 反向传播
# 自动计算梯度

loss.backward()

# 输出梯度
print("\nA 的梯度:")
print(A.grad)

print("\nB 的梯度:")
print(B.grad)
