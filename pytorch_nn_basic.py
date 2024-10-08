#%%
import torch

# 연산에 사용될 data type - 보통 계산을 하기 위한 데이터들에는 FloatTensor 사용
dtype = torch.FloatTensor

# GPU에서 Tensor를 연산할 경우
#dtype = torch.cuda.FloatTensor

batch_size, input_dim, hidden_dim, output_dim = 64, 1000, 100, 10

# 입력 데이터(x)와 출력 데이터(y) 무작위 생성
x = torch.randn(batch_size, input_dim).type(dtype)
y = torch.randn(batch_size, output_dim).type(dtype)

# 가중치 무작위 초기화
w1 = torch.randn(input_dim, hidden_dim).type(dtype)
w2 = torch.randn(hidden_dim, output_dim).type(dtype)

learning_rate = 1e-6
for t in range(500):
    # forward propagation
    # calculate prediction y
    # torch.mm => matrix multiplicaiton
    hidden = x.mm(w1)
    # relu 계산을 위해 min=0으로 지정한 값보다 작은 값들을 0으로 교체
    hidden_after_relu = hidden.clamp(min=0)
    # y prediction 값 도출
    y_pred = hidden_after_relu.mm(w2)

    # Loss(손실)값 계산
    loss = (y_pred - y).pow(2).sum()
    print(t, loss)

    # Loss값에 따른 w1, w2의 gradient를 계산하고 역전파
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = hidden_after_relu.t().mm(grad_y_pred)
    grad_hidden_after_relu = grad_y_pred.mm(w2.t())
    grad_hidden = grad_hidden_after_relu.clone()
    grad_hidden[hidden < 0] = 0
    grad_w1 = x.t().mm(grad_hidden)

    # Gradient Descent를 사용하여 가중치를 갱신
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
# %%
