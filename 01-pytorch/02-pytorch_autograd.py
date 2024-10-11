#%%

import torch
from torch.autograd import Variable

dtype = torch.FloatTensor
#dtype = torch.cuda.FloatTensor

batch_size, input_dim, hidden_dim, output_dim = 64, 1000, 100, 10

# input(x)와 출력(y)을 지정 -> 입력과 출력은 역전파 중 gradient를 계산할 필요가 없으므로 False로 지정
x= Variable(torch.randn(batch_size, input_dim).type(dtype), requires_grad=False)
y = Variable(torch.randn(batch_size, output_dim).type(dtype), requires_grad=False)

# 가중치 w1, w2는 역전파를 통해 gradient 값에 의해 업데이트됨 => True로 지정
w1 = Variable(torch.randn(input_dim, hidden_dim).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(hidden_dim, output_dim).type(dtype), requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # 역전파 단계를 별도로 구현하지 않기 위해 중간값들에 대한 참조(reference)를 유지
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    loss = (y_pred - y).pow(2).sum()
    print(t, loss.data[0])

    # autograd를 사용하여 역전파 수행
    # requires_grad=True를 지정한 모든 Variable에 대한 Loss의 gradient를 계산
    loss.backward()

    # w1.data, w1.grad.data = Tensor
    # w1.grad = Variable
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    # 가중치 갱신 후에는 수동으로 gradient를 0으로 만들어야 함
    w1.grad.data.zero_()
    w2.grad.data.zero_()
# %%
