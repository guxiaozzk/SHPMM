import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split
import jieba
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.utils import np_utils, to_categorical
from tensorflow.python.keras.utils import np_utils, to_categorical
import warnings
# df_data2 = pd.read_csv('../data/erc_train.csv',encoding='utf-8',sep='\t')

# df_data2 = pd.read_csv(r'D:\PycharmProjects\test\data\pdasentiment.csv',encoding='utf-8')
df_data2 = pd.read_csv(r'..\data\colthingsentiment.csv',encoding='utf-8',error_bad_lines= False)
# df_data2 = pd.read_csv(r'..\data\weibo_senti_100k.csv',encoding='utf-8',error_bad_lines= False)
df_data2 = df_data2.dropna()
df_data2 = df_data2.drop_duplicates()
df_data2 = df_data2.reset_index(drop=True)
print(len(df_data2))
##降采样（不平衡）


## word2vec 耗内存
from sklearn.utils import shuffle
data = df_data2
# col_unique = data['label'].unique()
# data_pos = data[data['label'].isin(['1'])]
# data_neg = data[data['label'].isin(['0'])]
# data_neg = shuffle(data_neg)
# data_pos = shuffle(data_pos)
# data_pos = data_pos.reset_index()
# data_neg = data_neg.reset_index()
# # data_neg = [i for i in range(len(data))  if data['label'][i]==0]
# # data_pos = data.loc[:,'label'==1]
# shape = min(data_pos.shape[0], data_neg.shape[0]) - 1
# # X_trainpos = X_train[X_train[['label']]==1]
# # data_neg = shuffle(data_neg)

# data_neg = data_neg.loc[0:shape, :]

# data_pos = data_pos.loc[0:shape, :]

# print(data_neg.shape[0])
# print(data_pos.shape[0])
# data = pd.concat([data_neg, data_pos], axis=0)
data = data.dropna()
df_data2 =data
df_data2 = df_data2.reset_index(drop=True)
print(len(df_data2))


##数据降噪需要针对已有模型的预测结果去降噪
## 这里做一些操作准备序列化文本数据来适应模型
def par_to_clean_cuttxt(content_lines):
    sentences=[]
    all_words=[]
    with open("C:\\Users\\guxiao\\Desktop\\数据集\\停顿词.csv", 'r', encoding='utf-8') as f:
        stop_words = f.read()
    def get_chinses(blog):
        if (blog >= u'\u4e00' and blog <= u'\u9fa5'):
            return True
    all_words=[]
    for line in content_lines:

        try:
            segs=jieba.lcut(str(line))
            #if (blog >= u'\u4e00' and blog <= u'\u9fa5')
            sent=[]
            for word in segs:
                # if word not in stop_words:
                    if get_chinses(word):
                        sent.append(word)
            segs=sent
            #print(segs)
            segs = filter(lambda x:len(x)>1, segs)
            segs = [v for v in segs if not str(v).isdigit()]#去数字
            segs = list(filter(lambda x:x.strip(), segs)) #去左右空格
            # segs = filter(lambda x:x not in stop_words, segs)
            #print(segs[0])
            #print("类型是")
            temp=[]

            for i in segs:
                all_words.append(i)
                temp.append(i)
            #print(all_words)
            # temp = ",".join(temp)
            #清洗后不分词的评论就是上面这样的
            #temp = " ".join(segs)
            # print("temp",temp)
            # print(type(temp))
            # if(len(temp)>1):
                # print(temp[0])
            # print(temp)
            sentences.append(temp)

        except Exception as e:
            print(e)
            continue
    # print(all_words)
    # print(sentences[0])
    # print(len(sentences))
    return sentences,all_words
maxlen = 44
#句子序列化 初始tokenizer
def seq_it(sentences):
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',oov_token ='U')
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    seq = tokenizer.texts_to_sequences(sentences)
    paded = pad_sequences(seq , maxlen = maxlen,truncating = "post",padding = "post")
    return paded,word_index

sents = list(df_data2['text'])
word_list = par_to_clean_cuttxt(sents)[0]

vocab = seq_it(word_list)[1]
print(len(word_list))

tok_padded= seq_it(word_list)[0]
#df_data2
labencoder =LabelEncoder()
df_data2['label']= labencoder.fit_transform(df_data2.label.values)  # label自己给的0 1 2

target_u=to_categorical(df_data2["label"],num_classes=6)
print(target_u)

x_train,x_test,y_train,y_test=train_test_split(tok_padded,target_u,random_state=1234,test_size=0.2)


##
# 在此基础之上修改产生CNN的遗传优化
from tensorflow.python import keras
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pylab as plt
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input,Embedding,Activation,Conv1D,GlobalMaxPooling1D,Flatten
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, Activation, GlobalMaxPooling1D, Dense,Input,Bidirectional,LSTM,Flatten,Permute,Reshape,Multiply
from tensorflow.keras import optimizers, losses, metrics, models
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.utils import np_utils, to_categorical

# 定义LSTM函数
def create_lstm(inputs, units, return_sequences):
    '''
    定义LSTM函数
    :param inputs:输入，如果这一层是第一层LSTM层，则传入layers.Input()的变量名，否则传入的是上一个LSTM层
    :param units: LSTM层的神经元
    :param return_sequences: 如果不是最后一层LSTM，都需要保留所有输出以传入下一LSTM层
    :return:
    '''
    lstm = keras.layers.Bidirectional(
        LSTM(units, return_sequences=return_sequences, dropout=0.3, kernel_regularizer=keras.regularizers.l2(0.001)))(
        inputs)
    #     lstm = keras.layers.Bidirectional(LSTM(units, return_sequences=return_sequences,dropout=0.3))(inputs)
    print('LSTM: ', lstm.shape)
    return lstm


def create_cnn(inputs, units):
    lstm = keras.layers.Conv1D(
        units,
        kernel_size=3,
        padding='same',
        activation='relu',
    )(inputs)
    print('CNN: ', lstm.shape)
    return lstm


def attention_3d_block(inputs, SINGLE_ATTENTION_VECTOR=False, stamp=44):
    input_dim = int(inputs.shape[2])  # shape = (batch_size, time_steps, input_dim)
    print('input_dim', input_dim)
    a = Permute((2, 1))(inputs)  # shape = (batch_size, input_dim, time_steps)

    a = Reshape((input_dim, stamp))(a)  # this line is not useful. It's just to know which dimension is what.
    #     a=inputs
    a = Dense(stamp, activation='softmax')(a)  # 为了让输出的维数和时间序列数相同（这样才能生成各个时间点的注意力值）

    a_probs = Permute((2, 1), name='attention_vec')(a)  # shape = (batch_size, time_steps, input_dim)
    print('att_vecoutput', a_probs.shape)

    output_attention_mul = Multiply()([inputs, a_probs])  # 把注意力值和输入按位相乘，权重乘以输入

    return output_attention_mul


def create_dense(inputs, units):
    '''
    定义Dense层函数
    :param inputs:输入，如果这一连接层是第一层全连接层，则需传入layers.Flatten()的变量名
    :param units: 全连接层单元数
    :return: 全连接层，BN层，dropout层
    '''
    # dense层
    dense = Dense(units, kernel_regularizer=keras.regularizers.l2(0.001))(inputs)
    #     dense = Dense(units)(inputs)
    #     print('Dense:', dense.shape)
    # dropout层
    dense_dropout = Dropout(rate=0.2)(dense)

    dense_batch = BatchNormalization()(dense_dropout)
    print('Dense:', dense_batch.shape)
    #     return dense, dense_dropout, dense_batch
    return dense, dense_dropout


def load():
    '''
    数据集加载
    :return:
    '''
    #     (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    #     # 数据集归一化
    #     x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test, y_train, y_test = train_test_split(tok_padded, target_u, random_state=1234, test_size=0.1)
    return x_train, y_train, x_test, y_test


def classify(x_train, y_train, x_test, y_test, num, epochs=5):
    '''
    利用num及上面定义的层，构建模型
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param num: 需要优化的参数(LSTM和全连接层层数以及每层神经元的个数)，同时，也是遗传算法中的染色体
    :return:
    '''
    # 设置LSTM层参数
    lstm_num_layers = num[0]
    lstm_units = num[2:2 + lstm_num_layers]
    lstm_name = list(np.zeros((lstm_num_layers,)))

    # 设置LSTM_Dense层的参数
    lstm_dense_num_layers = num[1]
    lstm_dense_units = num[2 + lstm_num_layers: 2 + lstm_num_layers + lstm_dense_num_layers]
    lstm_dense_name = list(np.zeros((lstm_dense_num_layers,)))
    lstm_dense_dropout_name = list(np.zeros((lstm_dense_num_layers,)))
    lstm_dense_batch_name = list(np.zeros((lstm_dense_num_layers,)))

    #     inputs_lstm = Input(shape=(x_train.shape[1], x_train.shape[2]))
    inputs_lstm = Input(shape=(maxlen,))

    #                         weights=[embedding_matrixzhihu],
    # embedding_dims,

    for i in range(lstm_num_layers):
        if i == 0:
            #             inputs = inputs_lstm
            #             print(inputs.shape)
            inputs = Embedding(input_dim=len(vocab) + 1, output_dim=100)(inputs_lstm)
            inputs = attention_3d_block(inputs)

            print(inputs.shape)
        else:
            inputs = lstm_name[i - 1]
        if i == lstm_num_layers - 1:
            return_sequences = False
        else:
            return_sequences = True

        lstm_name[i] = create_lstm(inputs, lstm_units[i], return_sequences=return_sequences)
    #         lstm_name[i] = create_cnn(inputs,lstm_units[i])

    for i in range(lstm_dense_num_layers):
        if i == 0:
            inputs = lstm_name[lstm_num_layers - 1]
        else:
            inputs = lstm_dense_name[i - 1]

        #         lstm_dense_name[i], lstm_dense_dropout_name[i], lstm_dense_batch_name[i] = create_dense(inputs,  units=lstm_dense_units[i])
        lstm_dense_name[i], lstm_dense_batch_name[i] = create_dense(inputs, units=lstm_dense_units[i])

    outputs_lstm = Dense(6, activation='softmax')(Flatten()(lstm_dense_batch_name[lstm_dense_num_layers - 1]))
    #     outputs_lstm = Dense(2, activation='softmax')(lstm_dense_batch_name[lstm_dense_num_layers - 1])
    # 维度不对是因为没有拉伸flatten dense之前要flatten

    # 构建模型
    LSTM_model = keras.Model(inputs=inputs_lstm, outputs=outputs_lstm, name='lstmModel')
    # 编译模型
    LSTM_model.compile(
        # optimizer=optimizers.Adam(lr=0.001),
                               optimizer=optimizers.RMSprop(lr=0.01),
        # loss='binary_crossentropy',
        loss = 'categorical_crossentropy',
        metrics=['accuracy'])

    history = LSTM_model.fit(x_train, y_train,
                             #                              batch_size=32, epochs=epochs, validation_split=0.1,verbose=1)
                             batch_size=64, epochs=epochs, validation_data=(x_test, y_test), verbose=1)
    # 验证模型
    results = LSTM_model.evaluate(x_test, y_test, verbose=0)
    return results[1], history  # 返回测试集的准确率


'''
    在优化神经网络上，用常规的遗传算法不易实现
    原因如下：
        1.传统的遗传算法中每条染色体的长度相同，但是优化LSTM网络时染色体的长度会因为层数的不同而不同
          比如：a染色体有一层LSTM层和一层全连接层，则这个染色体上共有6个基因(两个代表层数，两个代表神经元个数)
               b染色体有二层LSTM层和二层全连接层，则这个染色体上共有6个基因(两个代表层数，四个代表每层的神经元个数)

        2.在传统的遗传算法中，染色体上的基因的取值范围是相同的，但是在优化LSTM网络时，需要让表示层数的基因在一个范围内，
          表示神经元个数的基因在另一个范围内，比如层数范围是一到三层，神经元个数是32到256个之间
        3.由于染色体长度不同，交叉函数、变异函数均需要做出修改

    解决办法：
        1.将每条染色体设置为相同的长度
          (本题来说，因为LSTM层与全连接层层数最多三层，加上最前面两个表示层数的基因，故每条染色体上有3+3+2 = 8个基因)，
          达不到长度要求的后面补零
        2.先设置前面两个基因，令其范围分别在一到三之间，然后根据这两个基因确定后面关于每层神经元个数的基因
        3.对于交叉函数的修改，首先确定取出的两条染色体(设为a染色体和b染色体)上需要交换的位置，然后遍历两条染色体在这些位置的
          基因，如果任一染色体上此位置上的基因为0或要交换的基因是关于层数的，则取消此位置的交换
        4.对于变异函数的修改，只有关于神经元个数的基因变异，关于层数的基因不变异
'''

import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DNA_size = 2
DNA_size_max = 8  # 每条染色体的长度
POP_size = 3  # 种群数量
CROSS_RATE = 0.5  # 交叉率
MUTATION_RATE = 0.01  # 变异率
N_GENERATIONS = 3  # 迭代次数

x_train, y_train, x_test, y_test = load()


def get_fitness(x):
    return classify(x_train, y_train, x_test, y_test, num=x)[0]


def select(pop, fitness):
    idx = np.random.choice(np.arange(POP_size), size=POP_size, replace=True, p=fitness / fitness.sum())
    return pop[idx]


def crossover(parent, pop):
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_size, size=1)  # 染色体的序号
        cross_points = np.random.randint(0, 2, size=DNA_size_max).astype(np.bool)  # 用True、False表示是否置换

        # 对此位置上基因为0或是要交换的基因是关于层数的，则取消置换
        for i, point in enumerate(cross_points):
            if point == True and pop[i_, i] * parent[i] == 0:
                cross_points[i] = False
            # 修改关于层数的
            if point == True and i < 2:
                cross_points[i] = False
        # 将第i_条染色体上对应位置的基因置换到parent染色体上
        parent[cross_points] = pop[i_, cross_points]
    return parent


# 定义变异函数
def mutate(child):
    for point in range(DNA_size_max):
        if np.random.rand() < MUTATION_RATE:
            if point >= 3:
                if child[point] != 0:
                    child[point] = np.random.randint(32, 257)
    return child


# 层数
pop_layers = np.zeros((POP_size, DNA_size), np.int32)
pop_layers[:, 0] = np.random.randint(1, 4, size=(POP_size,))
pop_layers[:, 1] = np.random.randint(1, 4, size=(POP_size,))

# 种群
pop = np.zeros((POP_size, DNA_size_max))
# 神经元个数
for i in range(POP_size):
    pop_neurons = np.random.randint(8, 257, size=(pop_layers[i].sum(),))
    pop_stack = np.hstack((pop_layers[i], pop_neurons))
    for j, gene in enumerate(pop_stack):
        pop[i][j] = gene

# 迭代次数
for each_generation in range(N_GENERATIONS):
    # 适应度
    fitness = np.zeros([POP_size, ])
    # 第i个染色体
    for i in range(POP_size):
        pop_list = list(pop[i])
        # 第i个染色体上的基因
        # 将0去掉并变整数
        for j, each in enumerate(pop_list):
            if each == 0.0:
                index = j
                pop_list = pop_list[:j]
        for k, each in enumerate(pop_list):
            each_int = int(each)
            pop_list[k] = each_int

        fitness[i] = get_fitness(pop_list)
        print('第%d代第%d个染色体的适应度为%f' % (each_generation + 1, i + 1, fitness[i]))
        print('此染色体为：', pop_list)
    print('Generation:', each_generation + 1, 'Most fitted DNA:', pop[np.argmax(fitness), :], '适应度为：',
          fitness[np.argmax(fitness)])

    # 生成新的种群
    pop = select(pop, fitness)

    # 新的种群
    pop_copy = pop.copy()

    for i, parent in enumerate(pop):
        child = crossover(parent, pop_copy)
        child = mutate(child)
        pop[i] = child


