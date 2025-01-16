import numpy as np
import sys

from classes.Embedding import Embedding
from classes.SGD import SGD
from classes.Tensor import Tensor
from classes.cells import LSTMCell
from classes.losses import CrossEntropyLoss

np.random.seed(0)
# читаем файл
f = open('data/shakespear.txt', 'r')
raw = f.read()
f.close()

vocab = list(set(raw))  # список символов
word2index = {}
for i, word in enumerate(vocab):
    word2index[word] = i  # кажому символу сопоставляем индекс
indices = np.array(list(map(lambda x: word2index[x], raw)))  # представляем символы текста через индексы

embed = Embedding(vocab_size=len(vocab), dim=512)  # задаем эмбеддинг
model = LSTMCell(n_inputs=512, n_hidden=512, n_output=len(vocab))  # LSTM сеть с размерностью скрытого состояния 512
model.w_ho.weight.data *= 0

criterion = CrossEntropyLoss()
optim = SGD(parameters=model.get_parameters() + embed.get_parameters(),
            alpha=0.05)  # оптимизатор - стох. градиентный спуск

batch_size = 16
bptt = 25  # граница усечения (количество шагов обратного распространения)
n_batches = int((indices.shape[0] / (batch_size)))  # число батчей

trimmed_indices = indices[:n_batches * batch_size]  # сколько индексов поместится в батчи
batched_indices = trimmed_indices.reshape(batch_size, n_batches).transpose()

input_batched_indices = batched_indices[0:-1]
target_batched_indices = batched_indices[1:]

n_bptt = int(((n_batches - 1) / bptt))
input_batches = input_batched_indices[:n_bptt * bptt].reshape(n_bptt, bptt, batch_size)
target_batches = target_batched_indices[:n_bptt * bptt].reshape(n_bptt, bptt, batch_size)


def train(iterations=400):
    min_loss = 1000
    for iter in range(iterations):
        total_loss, n_loss = (0, 0)

        hidden = model.init_hidden(batch_size=batch_size)  # скрытый слой
        for batch_i in range(len(input_batches)):

            hidden = (Tensor(hidden[0].data, autograd=True),
                      Tensor(hidden[1].data, autograd=True))  # в отличие от RNN тут два скрытых вектора
            loss = None
            losses = list()
            for t in range(bptt):
                input = Tensor(input_batches[batch_i][t], autograd=True)
                rnn_input = embed.forward(input=input)
                output, hidden = model.forward(input=rnn_input, hidden=hidden)  # прямое распространение

                target = Tensor(target_batches[batch_i][t], autograd=True)
                # считаем потери
                batch_loss = criterion.forward(output, target)
                losses.append(batch_loss)
                if t == 0:
                    loss = batch_loss
                else:
                    loss = loss + batch_loss
            loss = losses[-1]

            # обратный ход
            loss.backward()
            optim.step()

            total_loss += loss.data / bptt
            epoch_loss = np.exp(total_loss / (batch_i + 1))

            if epoch_loss < min_loss:
                min_loss = epoch_loss
                print()
            log = "\r Iter:" + str(iter)
            log += " - Batch " + str(batch_i + 1) + "/" + str(len(input_batches))
            log += " - Min Loss:" + str(min_loss)[0:5]
            log += " - Loss:" + str(np.exp(total_loss / (batch_i + 1)))
            # if (batch_i == 0):
            log += " - " + generate_sample(n=70, init_char='\n').replace("\n", " ")

            sys.stdout.write(log)
        optim.alpha *= 0.99
        print()


def generate_sample(n=30, init_char=' '):
    s = ""
    hidden = model.init_hidden(batch_size=1)
    input = Tensor(np.array([word2index[init_char]]))
    for i in range(n):
        rnn_input = embed.forward(input)
        output, hidden = model.forward(input=rnn_input, hidden=hidden)
        output.data *= 15
        temp_dist = output.softmax()
        temp_dist /= temp_dist.sum()

        m = output.data.argmax()
        c = vocab[m]
        input = Tensor(np.array([m]))
        s += c
    return s


train(100)
print(generate_sample(n=2000, init_char='\n'))
