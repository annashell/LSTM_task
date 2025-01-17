"""
Классификатор текстов писателей на LSTM, попробовать GRU
"""
import glob
import os
import re
import sys

import numpy as np

from classes.Embedding import Embedding
from classes.SGD import SGD
from classes.Tensor import Tensor
from classes.cells import LSTMCell, RNNCell
from classes.losses import CrossEntropyLoss

np.random.seed(0)


def readText(fileName):  # функция принимает имя файла
    f = open(fileName, 'r', encoding="UTF-8")  # задаем открытие нужного файла в режиме чтения
    text = f.read()  # читаем текст
    text = text.replace("\n", " ")  # переносы строки переводим в пробелы
    text = re.sub('[–—!"#$%&«»…()*+,-./:;<=>?@[\\]^_`{|}~\t\n\xa0–\ufeff]', ' ', text)

    # return text.lower()[:(len(text) // 2)]  # функция возвращает текст файла
    return text.lower()


###########################
# Формирование обучающей выборки по листу индексов слов
# (разделение на короткие векторы)
##########################
def getSetFromIndexes(wordIndexes, xLen, step):
    xSample = []
    wordsLen = len(wordIndexes)
    index = 0

    # Идём по всей длине вектора индексов
    # "Откусываем" векторы длины xLen и смещаемся вперёд на step

    while index + xLen <= wordsLen:
        xSample.append(wordIndexes[index:index + xLen])
        index += step

    return xSample


def to_categorical(num, n_classes):
    return [int(num == i) for i in range(n_classes)]


###########################
# Формирование обучающей и проверочной выборки
# Из двух листов индексов от двух классов
##########################
def createSetsMultiClasses(wordIndexes, xLen,
                           step):  # функция принимает последовательность индексов, размер окна, шаг окна

    # Для каждого из 6 классов
    # Создаём обучающую/проверочную выборку из индексов
    nClasses = len(wordIndexes)  # задаем количество классов выборки
    classesXSamples = []  # здесь будет список размером "кол-во классов*кол-во окон в тексте*длину окна(например 6 по 1341*1000)"
    for wI in wordIndexes:  # для каждого текста выборки из последовательности индексов
        classesXSamples.append(getSetFromIndexes(wI, xLen,
                                                 step))  # добавляем в список очередной текст индексов, разбитый на "кол-во окон*длину окна"

    # Формируем один общий xSamples
    xSamples = []  # здесь будет список размером "суммарное кол-во окон во всех текстах*длину окна(например 15779*1000)"
    ySamples = []  # здесь будет список размером "суммарное кол-во окон во всех текстах*вектор длиной 6"

    for t in range(nClasses):  # в диапазоне кол-ва классов(6)
        xT = classesXSamples[t]  # берем очередной текст вида "кол-во окон в тексте*длину окна"(например 1341*1000)
        for i in range(len(xT)):  # и каждое его окно
            xSamples.append(xT[i])  # добавляем в общий список выборки

        # Формируем ySamples по номеру класса
        currY = to_categorical(t, xLen)  # текущий класс переводится в вектор длиной 6 вида [0.0.0.1.0.0.]
        for i in range(len(xT)):  # на каждое окно выборки
            ySamples.append(currY)  # добавляем соответствующий вектор класса

    xSamples = np.array(xSamples)  # переводим в массив numpy для подачи в нейронку
    ySamples = np.array(ySamples)  # переводим в массив numpy для подачи в нейронку

    return xSamples, ySamples  # функция возвращает выборку и соответствующие векторы классов


def prepare_data(data_folder, xLen, step, max_words):
    txt_files = glob.glob(f"{data_folder}/*.txt", recursive=True)

    # Загружаем обучающие тексты
    trainText = []
    train_files = [file_name for file_name in txt_files if "Обучающая" in file_name]
    for file_name in train_files:
        trainText.append(readText(file_name))

    # Загружаем тестовые тексты
    testText = []
    test_files = [file_name for file_name in txt_files if "Тестовая" in file_name]
    for file_name in test_files:
        testText.append(readText(file_name))

    className = ["Булгаков", "Саймак", "Фрай", "О. Генри", "Брэдбери", "Стругацкие"]
    nClasses = len(className)

    # разбиваем тексты на токены
    tokens = list(map(lambda x: x.split(" "), trainText))
    # составляем словарь, сортированный по возрастанию (т.к использовать будем не весь)
    vocab_dict = dict()
    for sent in tokens:
        for word in sent:
            if len(word) > 0:
                if word in vocab_dict.keys():
                    vocab_dict[word] += 1
                else:
                    vocab_dict[word] = 1

    sorted_vocab_dict = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)
    vocab = [word for word, count in sorted_vocab_dict]

    # Переводим слова в индексы
    word2index = {}
    for i, word in enumerate(vocab):
        if i < max_words - 1:
            word2index[word] = i
        else:
            word2index[word] = max_words - 1 # ограничиваем словарь при обучении

    # Переводим тексты в индексы
    input_dataset = list()
    for sent in tokens:
        sent_indices = list()
        for word in sent:
            if len(word) > 0:
                try:
                    sent_indices.append(word2index[word])
                except:
                    ""
        input_dataset.append(sent_indices)
        print(sent[:20], sent_indices[:20])

    # Переводим тексты в индексы для тестовых выборок
    test_dataset = list()
    tokens_test = list(map(lambda x: x.split(" "), testText))
    for sent in tokens_test:
        sent_indices = list()
        for word in sent:
            if len(word) > 0:
                try:
                    sent_indices.append(word2index[word])
                except:
                    ""
        test_dataset.append(sent_indices)

    print("Статистика по обучающим текстам:")
    len_train_text = 0
    len_train_dict = 0
    for i in range(nClasses):
        print(className[i], " ", len(trainText[i]), " символов, ", len(input_dataset[i]), " слов")
        len_train_text += len(trainText[i])
        len_train_dict += len(input_dataset[i])
    print("В сумме ", len_train_text, " символов, ", len_train_dict, " слов")
    print()

    print("Статистика по тестовым текстам:")
    len_test_text = 0
    len_test_dict = 0
    for i in range(nClasses):
        print(className[i], " ", len(testText[i]), " символов, ", len(test_dataset[i]), " слов")
        len_test_text += len(testText[i])
        len_test_dict += len(test_dataset[i])
    print("В сумме ", len_test_text, " символов, ", len_test_dict, " слов")

    # Формируем обучающую и тестовую выборку
    xTrain, yTrain = createSetsMultiClasses(input_dataset, xLen, step)  # извлекаем обучающую выборку
    xTest, yTest = createSetsMultiClasses(test_dataset, xLen, step)  # извлекаем тестовую выборку
    print(xTrain.shape)
    print(yTrain.shape)
    print(xTest.shape)
    print(yTest.shape)

    return xTrain, yTrain, xTest, yTest, max_words


def generate_prediction(model, xTest, yTest, vocabSize, batch_size):
    embed = Embedding(vocab_size=vocabSize, dim=512)
    hidden = model.init_hidden(batch_size=1)
    total_right_answers = 0

    possible_vars = [[int(j == i) for j in range(batch_size)] for i in range(6)]

    for i, input_test in enumerate(xTest):
        input = Tensor(input_test)
        rnn_input = embed.forward(input)
        output, hidden = model.forward(input=rnn_input, hidden=hidden)
        output.data *= 15
        temp_dist = output.softmax()
        temp_dist /= temp_dist.sum()

        normalized_output_all = []
        for output_b in output.data:
            max_arg = max(output_b)
            normalized_output = [int(x == max_arg) for x in output_b]
            normalized_output_all.append(normalized_output)
        vars_count = [normalized_output_all.count(var) for var in possible_vars]
        m = vars_count.index(max(vars_count))

        total_right_answers += int(sum(possible_vars[m] == yTest[i]) == batch_size)
    print(f"\n Процент правильных ответов: {round(total_right_answers / len(xTest) * 100, 2)}%")


def train(model, batch_size, xTrain, yTrain, vocabSize, iterations=100):
    model.w_ho.weight.data *= 0
    embed = Embedding(vocab_size=vocabSize, dim=512)  # задаем эмбеддинг
    criterion = CategorialCrossEntropyLoss()
    optim = SGD(parameters=model.get_parameters() + embed.get_parameters(),
                alpha=0.05)  # оптимизатор - стох. градиентный спуск

    bptt = 25  # граница усечения (количество шагов обратного распространения)

    n_batches = xTrain.shape[0]  # число батчей

    n_bptt = int(((n_batches - 1) / bptt))
    input_batches = xTrain[:n_bptt * bptt].reshape(n_bptt, bptt, batch_size)
    target_batches = yTrain[:n_bptt * bptt].reshape(n_bptt, bptt, batch_size)

    min_loss = 1000
    for iter in range(iterations):
        total_loss, n_loss = (0, 0)

        hidden = model.init_hidden(batch_size=batch_size)  # скрытый слой
        for batch_i in range(len(input_batches)):

            # hidden = (Tensor(hidden[0].data, autograd=True),
            #           Tensor(hidden[1].data, autograd=True))  # в отличие от RNN тут два скрытых вектора
            hidden = Tensor(hidden.data, autograd=True)
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
                # print(f"bptt {t/bptt} % batch {batch_i} iteration {iter}")
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
            sys.stdout.write(log)
        optim.alpha *= 0.99
        print()
        generate_prediction(model, xTest, yTest, vocabSize, batch_size)
    return model


# Задаём базовые параметры
step = 100  # Шаг разбиения исходного текста на обучающие вектора
batch_size = 16  # Длина отрезка текста, по которой анализируем, в словах
max_words = 15000 # максимальное число слов для обучения

xTrain, yTrain, xTest, yTest, vocabSize = prepare_data(r"data/Тексты писателей", batch_size, step, max_words)
model = RNNCell(n_inputs=512, n_hidden=512, n_output=batch_size)  # LSTM сеть с размерностью скрытого состояния 512
model = train(model, batch_size, xTrain, yTrain, vocabSize)
generate_prediction(model, xTest, yTest, vocabSize, batch_size)

# TODO уменьшить выборки в 10 раз, категориальная кроссэнтропия, в конце вернуть лстм
