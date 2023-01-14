#from snownlp impot SnowNLP
from sklearn.model_selection import GridSearchCV, train_test_split
from snownlp import sentiment, SnowNLP


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sentences = ['I like you just so so','I like you a little','I like you','I like you very much','I hate you just so so','I hate you a little','I hate you','I hate you very much',]
analyzer = SentimentIntensityAnalyzer()
for sentence in sentences:
    vs = analyzer.polarity_scores(sentence)
    print("{:-<65} {}".format(sentence, str(vs)))

##snownlp 直接适应评论数据 其他文本极性可以自己训练
# print(SnowNLP(u'本本已收到，体验还是很好，功能方面我不了解，只看外观还是很不错很薄，很轻，也有质感。').sentiments)
sentiment.train(r'../mydataset/tsinghua.negative.gb.txt', r'../mydataset/tsinghua.positive.gb.txt')

sentiment.save('sentiment.marshal')
model_path = 'sentiment.marshal'
##数据 每行一个文本 就行了
# sentiment.train('neg.txt', 'pos.txt')
sentiment.save(model_path)
sentiment.load(model_path)
print()
sent = "王子洋真菜啊"
print("snownlp",SnowNLP(sent).sentiments)

print("中文词典训练",sentiment.classify(sent))
import pandas as pd
import  jieba
import jieba.analyse
import OpenHowNet
# hownet_dict = OpenHowNet.HowNetDict()
# OpenHowNet.download()
# OpenHowNet.download()
hownet_dict_advanced = OpenHowNet.HowNetDict()
hownet_dict_advanced.initialize_similarity_calculation()
print(hownet_dict_advanced.calculate_word_similarity("苹果", "梨"))
##基于情感词典进行分类
##使用hownet进行增强
neg = pd.read_csv('../mydataset/tsinghua.negative.gb.txt',names=['text'])
pos = pd.read_csv('../mydataset/tsinghua.positive.gb.txt',names=['text'])
# df = pd.read_csv(r"../mydataset/chinesesentimentdic.csv",sep= ",",encoding="utf-8")
df = pd.read_csv(r"../data/tinghuadic.csv",sep= ",",encoding="utf-8")
dfgou = pd.read_csv(r"../data/foudingdic.csv",names=['text'])
key = df['key'].values.tolist()
score = df['score'].values.tolist()
def getlabel(str,neg,pos):
    for i in range(len(pos)):
        # if hownet_dict_advanced.calculate_word_similarity(str, pos['text'][i]) >= 0.8:
        if str==pos['text'][i]:
            # print("pos")
            return 1
            break

        else:
            continue

    for i in range(len(neg)):
        # if hownet_dict_advanced.calculate_word_similarity(str, neg['text'][i]) >= 0.8:
        if str==neg['text'][i]:
            # print("neg")
            return -1
            break

        else:
            continue

    return 0

def getscore(line):
    #segs  = line
    segs = jieba.cut(line)  #分词
    # segs = jieba.analyse.extract_tags(line, topK=20, withWeight=True, allowPOS=('a', 'v', 'vn','m'))
    # score_list = [score[key.index(hownet_getsimiar(x, df))] for x in segs if (hownet_getsimiar(x, df) in key)]
    # segs = jieba.analyse.extract_tags(line, topK=15, withWeight=True, allowPOS=('a', 'v', 'vn'))
    score_list  = [score[key.index(x)] for x in segs if(x in key)]
    # score_list = [getlabel(x,neg,pos) for x in segs ]
    # print(score_list)
    return  sum(score_list)  #计算得分
def hownet_getsimiarfou(str,df):
    # print("shabi")
    flag=0
    for i in range(len(df)):
        # if hownet_dict_advanced.calculate_word_similarity(str,df['key'][i])>=0.:
        if df['text'][i]==str:
            # print(str)
            flag=1
            break
    # print(flag)
    return  flag
def getfou(line):
    segs = jieba.cut(line)
    fou=[]
    for x in segs:
        # print("shabi1")
        if  hownet_getsimiarfou(x,dfgou)>0:
            fou.append(x)
    return fou
def  getattribute(line):
    neg=0
    pos=0
    negs=[]
    poss=[]
    score_list=[]
    segs = jieba.cut(line) # 分词
    score_list = [score[key.index(y)] for y in segs if (y in key)]

    # for i in segs:
    #     print(i)
    # score_list = [score[key.index(hownet_getsimiar(x,df))] for x in segs if (hownet_getsimiar(x,df) in key)]

    # print(score_list)
    negs=[x for x in segs if (x in key and score[key.index(x)==-1])]
    poss=[x for x in segs if (x in key and score[key.index(x)==1])]
    # print(len(segs))


    for  it in score_list:
        if  int(it)==-1:
            neg+=1
        if int(it)==1:
            pos+=1
    # print(negs)
    # print(poss)
    return neg,pos,negs,poss
line = "今天天气很好，我很开心"
print(getscore(line))


line0 = "今天下雨，心情不太好。"
print(getattribute(line0))
print(getscore(line0))
sent2 = u'本本已收到，体验还是很好，功能方面我不了解，只看外观还是很不错很薄，很轻，也有质感。'
print(round(getscore(sent2),2))
df_data2 = pd.read_csv(r'../data/colthingsentiment.csv',encoding='utf-8',error_bad_lines= False)

# x_train, x_test, y_train, y_test = train_test_split(df_data2['text'], df_data2['label'], test_size=0.3,random_state=1234)#所以把原数据集分成训练集的测试集，咱们用sklearn自带的分割函数。

from sklearn.utils import shuffle
data = df_data2
col_unique = data['label'].unique()
data_pos = data[data['label'].isin([1])]
data_neg = data[data['label'].isin([0])]
data_neg = shuffle(data_neg)
data_pos = shuffle(data_pos)
data_pos = data_pos.reset_index(drop=True)
data_neg = data_neg.reset_index(drop=True)
# data_neg = [i for i in range(len(data))  if data['label'][i]==0]
# data_pos = data.loc[:,'label'==1]
# shape = min(data_pos.shape[0], data_neg.shape[0]) - 1
# X_trainpos = X_train[X_train[['label']]==1]

shape = 4999
#取1000均衡数据集
# data_neg = shuffle(data_neg)

data_neg = data_neg.loc[0:shape, :]

data_pos = data_pos.loc[0:shape, :]

print(data_neg.shape[0])
print(data_pos.shape[0])
data = pd.concat([data_neg, data_pos], axis=0)
data = data.dropna()
df_data2 =data
print(len(df_data2))
df_data2 = df_data2.dropna()
df_data2 = df_data2.drop_duplicates()
df_data2 = df_data2.reset_index(drop=True)
print(len(df_data2))

##测试
# df_data2=pd.read_csv("../data/coltst.csv")
df_data2['neg'] = [getattribute(str(i))[0] for i in df_data2['text']]

df_data2['pos'] = [getattribute(str(i))[1] for i in df_data2['text']]

df_data2['negs'] = [getattribute(str(i))[2] for i in df_data2['text']]
df_data2['poss'] = [getattribute(str(i))[3] for i in df_data2['text']]
df_data2['score'] = [getscore(str(i)) for i in df_data2['text'] ]
df_data2['fou'] = [ len(getfou(str(i))) for i in df_data2['text'] ]
#
# th=2
# df_data2['tinghua-score'] = [0 if i < th else 1 for i in df_data2['score']]
#
# df_data2['preture'] = ['True' if df_data2['tinghua-score'][i] == df_data2['label'][i] else 'False'  for i in range(len(df_data2))]
# print(df_data2[ df_data2['preture'].isin(['False'])].shape)
# df_data2['preture']  = ['True' if df_data2['preture'][i]=='False' and df_data2['fou'][i]>0  else df_data2['preture'][i] for i in range(len(df_data2)) ]

# df_data2 =df_data2[ df_data2['preture'].isin(['True'])]
# df_data2=df_data2.reset_index(drop=True)
# print(df_data2[ df_data2['preture'].isin(['True'])].shape)
# df_data2.to_csv("colthingdiccls10000.csv")
# print(df_data2.head())
##1000-796

print(round((1- 796/10000),4))
# for i in range(len(df_data2)):
#     scorelist.append(getscore(df_data2['text'][i]))
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, precision_recall_curve, auc, roc_curve, f1_score
import numpy as np
import matplotlib.pyplot as plt
def plot_threshold(thresholds):
    pre_acc=[]
    # f1_acc=[]
    # print(len(thresholds))
    for th in thresholds:
        # y_score = model.decision_function(X)
        # y_score = y_preob
        # y_predict = (y_score > th)
        # df_data2['s-nlp-score'] = [0 if sentiment.classify(str(i)) < th else 1 for i in df_data2['text']]
        df_data2['tinghua-score'] = [0 if i < th else 1 for i in df_data2['score']]
        #
        df_data2['preture'] = ['True' if df_data2['tinghua-score'][i] == df_data2['label'][i] else 'False' for i in
                               range(len(df_data2))]
        print(df_data2[df_data2['preture'].isin(['False'])].shape)
        df_data2['preture'] = [
            'True' if df_data2['preture'][i] == 'False' and df_data2['fou'][i] > 0 else df_data2['preture'][i] for i in
            range(len(df_data2))]

        # print(df_data2['preture'].value_counts())
        accuracy_score = df_data2['preture'].value_counts()[0] / len(df_data2)
        # y_predict  = model.predict(X)
        pre_acc.append(accuracy_score)
        # f1_acc.append(f1_score(y_label,y_predict,average="macro"))
    # plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xticks(np.arange(-5,5,1))
    plt.title("Acc&&F1 with threhold")
    plt.plot(thresholds,pre_acc, marker='o', color='b')
    # # plt.xticks(threshold)
    # plt.plot(thresholds,f1_acc,color='g')
    plt.show()
plot_threshold(np.arange(-5,5,1))