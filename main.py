import torch

A = torch.randn(2, 2, requires_grad=True)

B = torch.randn(2, 2, requires_grad=True)

print("Tensor A:")
print(A)

print("\nTensor B:")
print(B)

C = torch.matmul(A, B)

print("\n矩阵乘法结果 C:")
print(C)

loss = C.sum()

print("\nLoss:")
print(loss)

loss.backward()

print("\nA 的梯度:")
print(A.grad)

print("\nB 的梯度:")
print(B.grad)