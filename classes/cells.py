import numpy as np

from classes.Layer import Layer
from classes.Linear import Linear
from classes.Tensor import Tensor
from classes.activation import Sigmoid, Tanh


class RNNCell(Layer):

    def __init__(self, n_inputs, n_hidden, n_output, activation='sigmoid'):
        super().__init__()

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_output

        if (activation == 'sigmoid'):
            self.activation = Sigmoid()
        elif (activation == 'tanh'):
            self.activation == Tanh()
        else:
            raise Exception("Non-linearity not found")

        self.w_ih = Linear(n_inputs, n_hidden)
        self.w_hh = Linear(n_hidden, n_hidden)
        self.w_ho = Linear(n_hidden, n_output)

        self.parameters += self.w_ih.get_parameters()
        self.parameters += self.w_hh.get_parameters()
        self.parameters += self.w_ho.get_parameters()

    def forward(self, input, hidden):
        from_prev_hidden = self.w_hh.forward(hidden) # достаем информацию о предыдущих состояниях из скрытого вектора
        combined = self.w_ih.forward(input) + from_prev_hidden # добавляем агрегированную информацию о предыдущих состояниях
        new_hidden = self.activation.forward(combined) # обучаем на комбинации текущего вектора и информации о предыдущих
        output = self.w_ho.forward(new_hidden)
        return output, new_hidden

    def init_hidden(self, batch_size=1):
        return Tensor(np.zeros((batch_size, self.n_hidden)), autograd=True)


class LSTMCell(Layer):

    def __init__(self, n_inputs, n_hidden, n_output):
        super().__init__()

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.xf = Linear(n_inputs, n_hidden) # веса слоя вентиля забывания, умн. на вх. вектор
        self.xi = Linear(n_inputs, n_hidden) # веса слоя входного вентиля, умн. на вх. вектор
        self.xo = Linear(n_inputs, n_hidden) # веса слоя выходного вентиля, умн. на вх. вектор
        self.xc = Linear(n_inputs, n_hidden) # веса слоя вентиля обновления, умн. на вх. вектор

        self.hf = Linear(n_hidden, n_hidden, bias=False) # веса вектора скрытого состояния вентиля забывания
        self.hi = Linear(n_hidden, n_hidden, bias=False) # веса вектора скрытого состояния входного вентиля
        self.ho = Linear(n_hidden, n_hidden, bias=False) # веса вектора скрытого состояния выходного вентиля
        self.hc = Linear(n_hidden, n_hidden, bias=False) # веса вектора скрытого состояния вентиля обновления

        self.w_ho = Linear(n_hidden, n_output, bias=False) # веса

        self.parameters += self.xf.get_parameters()
        self.parameters += self.xi.get_parameters()
        self.parameters += self.xo.get_parameters()
        self.parameters += self.xc.get_parameters()

        self.parameters += self.hf.get_parameters()
        self.parameters += self.hi.get_parameters()
        self.parameters += self.ho.get_parameters()
        self.parameters += self.hc.get_parameters()

        self.parameters += self.w_ho.get_parameters()

    def forward(self, input, hidden):
        prev_hidden = hidden[0]
        prev_cell = hidden[1]

        f = (self.xf.forward(input) + self.hf.forward(prev_hidden)).sigmoid()  # вентиль забывания
        i = (self.xi.forward(input) + self.hi.forward(prev_hidden)).sigmoid()  # входной вентиль
        o = (self.xo.forward(input) + self.ho.forward(prev_hidden)).sigmoid()  # выходной вентиль
        u = (self.xc.forward(input) + self.hc.forward(prev_hidden)).tanh()  # вентиль обновления
        c = (f * prev_cell) + (i * u) # обновляем долгосрочную информацию

        h = o * c.tanh()  # скрытый слой

        output = self.w_ho.forward(h)
        return output, (h, c)

    def init_hidden(self, batch_size=1):
        init_hidden = Tensor(np.zeros((batch_size, self.n_hidden)), autograd=True)
        init_cell = Tensor(np.zeros((batch_size, self.n_hidden)), autograd=True)
        init_hidden.data[:, 0] += 1
        init_cell.data[:, 0] += 1
        return (init_hidden, init_cell)


class Dropout:
    def __init__(self, rate):
        self.rate = rate  # Вероятность "выброса" нейронов
        self.mask = None  # Маска для Dropout

    def forward(self, x, training=True):
        if training:
            # Генерируем маску: 1 - сохраняем нейрон, 0 - выбрасываем
            self.mask = np.random.binomial(1, 1 - self.rate, size=x.data.shape)
            x.data = x.data * self.mask
            return x # Применяем маску к входным данным
        else:
            return x  # Просто возвращаем входные данные

    def backward(self, dout):
        # Применяем маску к градиентам
        return dout * self.mask