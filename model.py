#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
import numpy as np
import jieba
import random
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import gensim.models.word2vec
from gensim.models import word2vec, KeyedVectors
import re
import numpy as np
import sklearn.naive_bayes as nb
import matplotlib.pyplot as mp
import sklearn.model_selection as ms
import sklearn.metrics as sm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.python import keras, Tensor
from tensorflow.python.keras import Sequential, regularizers
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Embedding, Dropout, Conv1D, Activation, GlobalMaxPooling1D, Dense,     Convolution2D, GlobalMaxPooling2D, BatchNormalization
from tensorflow.python.keras.optimizers import SGD, Adam,RMSprop
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.utils import np_utils, to_categorical
import warnings
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.keras.backend import set_session
warnings.filterwarnings("ignore")


# In[2]:


#文本预处理
file_path=open(r'C:\Users\guxiao\Desktop\数据集\停用词表.txt',encoding='utf-8')
stopwords=file_path.read()
df_data2 = pd.read_csv(r'D:\PycharmProjects\test\data\colthingdiccls10000.csv')
df_data2 = df_data2.dropna()
df_data2 = df_data2.drop_duplicates()
df_data2 = df_data2.reset_index(drop=True)



# In[3]:


#制作融合情感向量
data=df_data2
min_max_scaler = MinMaxScaler()
data = pd.concat([data['neg'],data['pos'],data['fou'],data['score']],axis=1)
#转成array
test_arr= min_max_scaler.fit_transform(data.values)


# In[4]:


##降采样（防止不平衡）
from sklearn.utils import shuffle
data = df_data2.reset_index(drop=True)
col_unique = data['label'].unique()
data_pos = data[data['label'].isin([1])]
data_neg = data[data['label'].isin([0])]
data_neg = shuffle(data_neg)
data_pos = shuffle(data_pos)
data_pos = data_pos.reset_index(drop=True)
data_neg = data_neg.reset_index(drop=True)
shape = min(data_pos.shape[0], data_neg.shape[0]) - 1
data_neg = data_neg.loc[0:shape, :]
data_pos = data_pos.loc[0:shape, :]
print(data_neg.shape[0])
print(data_pos.shape[0])
data = pd.concat([data_neg, data_pos], axis=0)
data = data.dropna()
df_data2 =data
print(len(df_data2))

#按标签分布（降采样结果）
plt.style.use('fivethirtyeight')
plt.title('newdata')
sns.countplot('label',data = df_data2)
plt.show()


# In[5]:


#文本预处理形成分词与总词词库
def par_to_clean_cuttxt(sentences):
    def get_chinses(blog):#过滤所有非中文字符
        if (blog >= u'\u4e00' and blog <= u'\u9fa5'):
            return True

    new_sents = sentences
    word_list=[]
    all_list=[]
    with open("C:\\Users\\guxiao\\Desktop\\数据集\\停顿词.csv", 'r', encoding='gbk') as f:
        stop_words = f.read()

    for sents in new_sents :
        sents =jieba.lcut(str(sents))
    #print(type(sents))
        sent=[]
        for word in sents :
                if get_chinses(word):
                    sent.append(word)
                    all_list.append(word)
        segs=sent
        segs = [v for v in segs if not str(v).isdigit()]#去数字
        segs = list(filter(lambda x:x.strip(), segs)) #去左右空格
        word_list.append(segs)
    print(len(word_list))
    print(word_list[0])
    return word_list,all_list


# In[6]:


#长度分布统计
clear_sentences=[]
sents = list(df_data2['text'])
word_list = par_to_clean_cuttxt(sents)[0]
worc_list = [ len(x) for x in word_list]
df_data2['sentence_len'] = worc_list
sns.distplot(df_data2['sentence_len'])
plt.show()


# In[7]:


# 取tokens平均值并加上两个tokens的标准差，
max_tokens = np.mean(df_data2["sentence_len"]) + 2 * np.std(df_data2["sentence_len"])
max_tokens = int(max_tokens)
print('max_tokens:', max_tokens)


# In[8]:


#Word2Vec模型构建
## 这个构建需要的数据集有一个特点就是需要二维列表
all_list = par_to_clean_cuttxt(sents)[1]
modelw = word2vec.Word2Vec(word_list,vector_size=100)
modelw.wv.init_sims(replace=True)
modelw.save('colmodel.model')
modelw = word2vec.Word2Vec.load('colmodel.model')

print(type(modelw))


# In[9]:


# 设置参数
maxlen = max_tokens
batch_size = 32
embedding_dims = maxlen
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 10
nclasses = 2


# In[10]:


##句子序列化 初始tokenizer
#加载Word2Vec的词向量构建函数
def seq_it(sentences):
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',oov_token ='U')
    # print(sentences)
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    # seq = tokenizer.texts_to_sequences(sentences)
    ##加载word2vec的序列化
    seq=[]
    for sent in sentences:
        for i, word in enumerate(sent):
            try:
                sent[i] = modelw.wv.key_to_index[word]
            except (KeyError,ValueError):
                sent[i] = 0
        seq.append(sent)

    paded = pad_sequences(seq , maxlen = maxlen,truncating = "pre",padding = "pre")
    return paded,word_index
tok_padded= seq_it(word_list)[0]
target_u=to_categorical(df_data2["label"],num_classes=2)


# In[11]:


##词袋模型构建
str_list = []#以逗号连接每一个词组
word_list = par_to_clean_cuttxt(sents)[0]
modelw = word2vec.Word2Vec(word_list,vector_size=100)
modelw.wv.init_sims(replace=True)
for it in word_list:
    str_list.append(",".join(it))
mx_train =np.array(str_list)
count = CountVectorizer(max_features=40000)
# tfidf2 = TfidfVectorizer(analyzer='word',ngram_range=(1,2),max_features=4000)
# tx_train= tfidf2.fit_transform(mx_train)
cx_train = count.fit_transform(mx_train)
x_train,x_test,y_train,y_test=train_test_split(cx_train,df_data2["label"],random_state=1234,test_size=0.1)


# In[12]:


X_train = x_train
X_test = x_test
svc = SVC(C=5,kernel='rbf',gamma=2.469,probability=True)
lr = LogisticRegression(random_state=42,penalty='l2',class_weight='balanced')
dt = DecisionTreeClassifier(max_depth=1, random_state=42)
rf = RandomForestClassifier(n_estimators=100, max_depth=1, random_state=42)
ab = AdaBoostClassifier(n_estimators=200, random_state=42)
log_loss_list = [] # 用于存储各个模型的二分类交叉熵
for i in [svc,lr, dt, rf, ab]:
    lists=[]
    i.fit(X_train, y_train) # 训练模型
    y_presvm = i.predict(X_train)
    probs = accuracy_score(y_train, y_presvm)
    lists.append(probs)
    prob = i.predict_proba(X_test)  # 预测
    lists.append(round(log_loss(y_test, prob), 4))
    lists.append(round(accuracy_score(y_test,i.predict(X_test)),4))
    log_loss_list.append(lists)
former = pd.DataFrame(log_loss_list, index=[['支持向量机','逻辑回归', '决策树', '随机森林', 'AdaBoost']], columns=['拟合系数','分类交叉熵','预测系数'])
print(former)
cla= lr
cla.fit(x_train,y_train)
print(cla.score(x_train,y_train))
print(cla.score(x_test,y_test))
##评价模型
"""
    模型训练、预测完成后  可以统计分类结果的 混淆矩阵、分类报告
    混淆矩阵 : 矩阵  行：分类 列：分类
    分类报告 ：召回率 F1分数 等结果
"""
model = cla
test_y = y_test
test_x = x_test
prd_test_y = model.predict(test_x)
# 混淆矩阵 矩阵  行：分类 列：分类
cm = sm.confusion_matrix(test_y, prd_test_y)
print("---------------混淆矩阵\n", cm)

cp = sm.classification_report(test_y, prd_test_y,digits=4)
print("---------------分类报告\n", cp)
acc = np.sum(prd_test_y == test_y)/test_y.size
print(acc)



vocab = seq_it(word_list)[1]
##确定初始化的word2vec不为空
embeding_matrix=np.zeros((len(vocab)+1,100))
print(embeding_matrix)
for word,i in vocab.items():
    try:
        embeding_vector=modelw.wv[str(word)]
        embeding_matrix[i]=embeding_vector
    except KeyError:
        continue
zero= np.zeros(shape = embeding_matrix.shape)
##判断初始wordmatrix是否为空
print("为空吗",(zero==embeding_matrix).all())
# max_features = len(vocab)+1
##记住word2vec改变的是初始权重 不是别的 初始权重 embeding_matrix
#x_test = seq_it(x_test)[0]

#x_train = to_review_vector(x_train)
#x_test = to_review_vector(x_test)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('Build model...')


# In[ ]:





# In[ ]:





# In[14]:


# from sklearn.model_selection import GridSearchCV

# # 导入包
# from sklearn.metrics import precision_recall_curve
# from sklearn.model_selection import TimeSeriesSplit
# clf = LogisticRegression(class_weight='balanced')
# # clf = svc
# # param_grid = {'C':[1000,500,100,50,10,5,1,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001],'penalty':['l1','l2']} 
# param_grid ={'C':np.arange(0.1,5,0.1),'penalty':['l1','l2']}
# # param_grid = {'C':[1,5,10],'kernel':['linear', 'poly', 'rbf', 'sigmoid']}
# #SVM寻优
# tscv = TimeSeriesSplit(n_splits=10) 
# # gsv = GridSearchCV(clf,param_grid,cv=tscv, scoring = 'f1_micro', verbose=1, n_jobs = -1)
# gsv = GridSearchCV(clf,param_grid,cv=tscv, scoring = 'accuracy', verbose=1, n_jobs = -1)

# # f1是f1score吗 
# gsv.fit(x_train,y_train)

# print("Best HyperParameter: ",gsv.best_params_)



# In[15]:


#准备情感向量与词向量的并行输入
x_train,x_test,y_train,y_test=train_test_split(tok_padded,target_u,random_state=1234,test_size=0.1)
arr_train,arr_test,yarr_train,yarr_test=train_test_split(test_arr.reshape(test_arr.shape[0],test_arr.shape[1],1),target_u,random_state=1234,test_size=0.1)


# In[19]:


from tensorflow.python.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input,Embedding,Permute,Reshape,Multiply,Flatten
from tensorflow.python.keras.callbacks import ModelCheckpoint
import tensorflow as tf
def build_lstm_model(embedding_dims,maxlen):
    model = Sequential()

    model.add(Embedding(len(vocab)+1,
                        100,
                        weights=[embeding_matrix],
                        input_length=maxlen))
    model.add(LSTM(32, return_sequences=True,dropout=0.4))
    model.add(LSTM(32, return_sequences=True,dropout=0.4))
    model.add(keras.layers.Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(hidden_dims/2))
    model.add(Dense(nclasses))
    model.add(Activation('softmax'))
    initial_learning_rate=0.01
    adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model
##尝试做两种不同输入的融合
def merge_model():
    main_input = Input(shape=(maxlen,),name='main_input')
    aux_input = Input(shape=(4,1),name='aux_input')
    x = Embedding(len(vocab)+1,
                        100,
                        weights=[embeding_matrix],
                        input_length=maxlen
                 )(main_input)
    x = keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True,dropout=0.3))(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(16,dropout=0.3))(x)

    x = Flatten()(x)
    x = Dense(128, kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = Dropout(0.2)(x)
    out1 = BatchNormalization()(x)
    main_output = Dense(nclasses,activation='softmax',name='main_output')(out1)
    x = keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True))(aux_input)
    x = Flatten()(x)
    out2 = Dense(64, activation="relu")(x)

    aux_output = Dense(nclasses,activation='softmax',name='aux_output')(out2)
    #  融合输出
    x = keras.layers.Concatenate(axis= 1)([out1, out2])
    
    concatresult = Dense(nclasses,activation='softmax',name='concat')(x)
    model =keras.models.Model(inputs=[main_input, aux_input], outputs=concatresult)
    adam = Adam(lr= 0.0001)
    rmsprop = RMSprop(lr=0.0001)
    model.compile(optimizer=rmsprop,
                  
                loss='binary_crossentropy',
                metrics=['binary_accuracy'])
    return it,model
def build_bilstm_model(embedding_dims,maxlen):
    model = Sequential()

    model.add(Embedding(

        len(vocab)+1,
#                    
        100,weights = [embeding_matrix],

                        input_length=maxlen


    ))  
    model.add(Dropout(0.5))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(16, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(16)))
    model.add(keras.layers.Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(hidden_dims/4))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(nclasses))
    model.add(Activation('softmax'))
    opt = SGD(lr= 0.001,momentum=0.9)
    adam = Adam(lr= 0.0001)
    rmsprop = RMSprop(lr=0.0001)
    bin_acc = keras.metrics.BinaryAccuracy(name='binary_accuracy', dtype=None, threshold=0.2)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,# 这里曾经出现过0.96 是0.001 
                  metrics=['accuracy'])
    return model


# In[ ]:





# In[20]:


import os 
from tensorflow.python.keras.utils import np_utils, to_categorical, plot_model


# In[21]:


from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.keras.backend import set_session
import tensorflow as tf
tf.keras.backend.clear_session()  # For easy reset of notebook state.清空会话



config_proto = tf.ConfigProto()
off = rewriter_config_pb2.RewriterConfig.OFF
config_proto.graph_options.rewrite_options.arithmetic_optimization = off
session = tf.Session(config=config_proto)
set_session(session)
#callback修饰历史


#模型调用

it,model = merge_model()[0],merge_model()[1]
# os.environ["PATH"] += os.pathsep + r'D:\Graphviz\bin'
# plot_model(model, to_file=r'D:\PycharmProjects\test\result\mergemodel.png', show_shapes=True, show_layer_names=True)
# print("finissh 保存结构图")
model = build_model2(embedding_dims=embedding_dims,maxlen=maxlen)
units = 64
num_classes = 2
    #batch_size = 32

#model= RNN(units, num_classes, num_layers=2)
filepath = "bestlstm_weights.h5"
 
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max', period=1)
# callbacks_list = [checkpoint]
it = model.fit(x={'main_input': x_train, 'aux_input': arr_train},
                y={'concat':y_train},
#                 validation_split=0.1,
                validation_data=({'main_input': x_test, 'aux_input': arr_test}, {'concat':y_test}),
                batch_size=32, epochs=30,verbose=1)
# it = model.fit(
#                 x_train, y_train,
#               batch_size=64,
#               epochs=20,
#               validation_data=(x_test, y_test),                 
# #               callbacks = callbacks_list,
#                     shuffle=True
#                  )

# history.loss_plot("epoch")
##调试模型与损失可视化
import matplotlib.pyplot as plt
def plot_hoistory(history,string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("epoch")
    plt.ylabel(string)
    plt.legend(string, 'val_'+string)
    plt.show()
# print(model.evaluate(x_test, y_test))

# plot_hoistory(it,"acc")
# plot_hoistory(it,"loss")
# print(model.evaluate(x_test, y_test))

# import  matplotlib.pyplot as pyplot
# history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, verbose=0)
# # evaluate the model
#                      团会心得
#   此次团会的主题是砥砺前行，感恩母校，首先我们现在仍然是辽宁工程技术大学的一分子，无论是找到了工作已经签约，还是考上了研究生等待着复试，我们都是辽宁工程技术大学的一员，我们在未来将会获得更多，同样也会失去很多，我们将进一步地服务社会，或者去消费社会，但是无论如何，我们现在仍然是辽宁工程技术大学的组成因子，我们应该时刻秉持太阳石精神，在学校的最后一年依然争取做一个品学兼优，尊师爱幼的好学长或者好学姐，我们在离开学校之后便会留下一个深深的烙印，我们的第一学历是辽宁工程技术大学，我们在工作中还是在社会上都代表着辽宁工程技术大学的面貌与精神，我们是母校的招生牌，也是母校的口碑栏，我们在未来还是现在一定要以大学生的身份来要求自己，以共青团员的面貌来展示自己，这样才可以不负母校，不负四年求学
# # _, train_acc = model.evaluate(x_train ,y_train, verbose=0)
# _, test_acc = model.evaluate(x_test, y_test, verbose=0)
# print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# # plot loss during training
# pyplot.subplot(211)
# pyplot.title('Loss')
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# # plot accuracy during training
# pyplot.subplot(212)
# pyplot.title('Accuracy')
# pyplot.plot(history.history['accuracy'], label='train')
# pyplot.plot(history.history['val_accuracy'], label='test')
# pyplot.legend()
# pyplot.show()


# In[ ]:


# from tensorflow.core.protobuf import rewriter_config_pb2
# from tensorflow.keras.backend import set_session
import tensorflow.python.keras.callbacks
import tensorflow.python.keras.backend as K
import tensorflow as tf
# from tensorflow import keras
# tf.keras.backend.clear_session()  # For easy reset of notebook state.清空会话

# config_proto = tf.ConfigProto()
# off = rewriter_config_pb2.RewriterConfig.OFF
# config_proto.graph_options.rewrite_options.arithmetic_optimization = off
# session = tf.Session(config=config_proto)
# set_session(session
def temp(model):
    def scheduler(epoch):
        lr = K.get_value(model.optimizer.lr)
        if epoch == 10:
            lr *= 0.1
#         elif epoch == 11:
#             lr *= 0.1
#         elif epoch == 9:
#             lr *= 0.5
        elif epoch == 5:
            lr *= 0.1
        print(lr)
        return lr
    return scheduler
#模型调用

re_l = temp(model)
reduce_lr = keras.callbacks.LearningRateScheduler(re_l)

#model= RNN(units, num_classes, num_layers=2)
x_train,x_test,y_train,y_test=train_test_split(tok_padded,target_u,random_state=1234,test_size=0.1)
# checkpoint = ModelCheckpoint(
#      monitor = 'val_accuracy',
#     mode = 'max',
#     filepath = './input/LSTM.h5',
#     save_best_only=True,
#     verbose=1
# )
with tf.Session().as_default() as sess:
    sess.run(tf.global_variables_initializer())
    it = model.fit(x={'main_input': x_train, 'aux_input': arr_train},
                y={'concat':y_train},
#                 validation_split=0.1,
                validation_data=({'main_input': x_test, 'aux_input': arr_test}, {'concat':y_test}),
                batch_size=32, epochs=20,verbose=1,
#                           callbacks=[reduce_lr]
                  )

# history.loss_plot("epoch")
##调试模型与损失可视化

# plot_hoistory(it,"loss")
# print(model.evaluate(x_test, y_test))

# import  matplotlib.pyplot as pyplot
# history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, verbose=0)
# # evaluate the model
# # _, train_acc = model.evaluate(x_train ,y_train, verbose=0)
# _, test_acc = model.evaluate(x_test, y_test, verbose=0)
# print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# # plot loss during training
# pyplot.subplot(211)
# pyplot.title('Loss')
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# # plot accuracy during training
# pyplot.subplot(212)
# pyplot.title('Accuracy')
# pyplot.plot(history.history['accuracy'], label='train')
# pyplot.plot(history.history['val_accuracy'], label='test')
# pyplot.legend()
# pyplot.show()


# In[ ]:


import os
with tf.Session().as_default() as sess:
    print(model.evaluate({'main_input': x_test, 'aux_input': arr_test}, {'concat':y_test}))
# print(model.evaluate(x={'main_input': x_test, 'aux_input': arr_test}))


# In[ ]:


import seaborn as sn
import itertools
import matplotlib.pyplot as plt
import  numpy as np

test_pred = np.argmax(y_test,axis=1)
y_pre=np.argmax(model.predict({'main_input': x_test, 'aux_input': arr_test}),axis=1)
# print(model.evaluate(x_test,y_test))
sn.heatmap(confusion_matrix(y_pre, test_pred), annot=True)
plt.show()
	# annot=True，显示各个cell上的数字
cm = sm.confusion_matrix(y_pre, test_pred)
print("---------------混淆矩阵\n", cm)

cp = sm.classification_report(y_pre, test_pred,digits=4)
print("---------------分类报告\n", cp)
acc = round(np.sum(test_pred == y_pre)/len(y_test),4)
print(acc)

