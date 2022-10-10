#%%
import torch
import matplotlib.pyplot as plt
import snsplot
snsplot.set()

#%%
# 定义输入输出层的个数 和上面一样
N, D_in, H, D_out = 64, 1000, 100, 10

# 创造训练集
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
x = x.cuda()
y = y.cuda()

# 定义模型
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device =", device)
model.to(device)

# 如果训练效果不好，也可以加上这两句试试，深度学习有点玄学
#torch.nn.init.normal_(model[0].weight)
#torch.nn.init.normal_(model[1].weight)

# 开始神经网络的计算,但是这里我们使用优化器帮我们更新参数
#%%
learning_rate = 1e-6
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

it_ls = []
loss_ls = []
for it in range(15000):
    # 前向传播
    y_pred = model(x)

    # 计算损失
    loss = loss_fn(y_pred, y)
    print(it, loss.item())
    it_ls.append(it)
    loss_ls.append(loss.item())

    # 梯度清零, 然后反向传播
    optimizer.zero_grad()
    loss.backward()

    # 更新参数，这里只需要一句话
    optimizer.step()

#%%
fig, ax = plt.subplots()
ax.plot(it_ls, loss_ls)
ax.set_xlabel("step")
ax.set_ylabel("loss")
plt.show()
# %%
