#%%
import torch
import time
from torch import nn
#%%
print("GPU availability =", torch.cuda.is_available())
print("GPU amount:", torch.cuda.device_count())
print("GPU name:", torch.cuda.get_device_name(0))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%
# 准备数据
n = 1000000  # 样本数量

X = 10 * torch.rand([n, 2]) - 5.0  # torch.rand是均匀分布
w0 = torch.tensor([[2.0, -3.0]])
b0 = torch.tensor([[10.0]])
Y = X @ w0.t() + b0 + torch.normal(0.0, 2.0, size=[n, 1])  # @表示矩阵乘法,增加正态扰动
#%%
# 移动到GPU上
X = X.cuda()
Y = Y.cuda()

#%%
# 定义模型
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn_like(w0))
        self.b = nn.Parameter(torch.zeros_like(b0))

    # 正向传播
    def forward(self, x):
        return x @ self.w.t() + self.b
linear = LinearRegression()
#%%
# 移动模型到GPU上
linear.to(device)
#%%
# 训练模型
optimizer = torch.optim.Adam(linear.parameters(), lr=0.1)
loss_func = nn.MSELoss()

def train(epoches):
    tic = time.time()
    for epoch in range(epoches):
        optimizer.zero_grad()
        Y_pred = linear(X)
        loss = loss_func(Y_pred, Y)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print({"epoch": epoch, "loss": loss.item()})
    toc = time.time()
    print("time used:", toc - tic)

train(500)
# %%
