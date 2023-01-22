import logging
import re
import time
from datetime import timedelta
import numpy as np
from logging import handlers
import jieba
import pandas as pd
import sys
sys.path.append("../../src")
from utils import config
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn import metrics
def query_cut(query):
    return list(jieba.cut(query))

def get_score(Train_label, Test_label, Train_predict_label,
              Test_predict_label):
    '''
    @description: get model score
    @param {type}
    Train_label, ground truth label of train data set
    Test_label, ground truth label of test dataset
    @return:acc, f1_score
    '''
    # 输出模型的准确率， 精确率，召回率， f1_score
    return metrics.accuracy_score(
        Train_label, Train_predict_label), metrics.accuracy_score(
            Test_label, Test_predict_label), metrics.recall_score(
                Test_label, Test_predict_label,
                average='micro'), metrics.f1_score(Test_label,
                                                   Test_predict_label,
                                                   average='weighted')

def create_logger(log_path):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }
    
    logger = logging.getLogger(log_path)
    fmt = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    format_str = logging.Formatter(fmt)  # 设置日志格式
    logger.setLevel(level_relations.get('info'))  # 设置日志级别
    sh = logging.StreamHandler()  # 往屏幕上输出
    sh.setFormatter(format_str)  # 设置屏幕上显示的格式
    th = handlers.TimedRotatingFileHandler(
        filename=log_path, when='D', backupCount=3,
        encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
    th.setFormatter(format_str)  # 设置文件里写入的格式
    logger.addHandler(sh)  # 把对象加到logger里
    logger.addHandler(th)

    return logger

def wam(sentence, w2v_model, method='mean', aggregate=True):
    '''
    @description: 通过word average model 生成句向量
    @param {type}
    sentence: 以空格分割的句子
    w2v_model: word2vec模型
    method： 聚合方法 mean 或者max
    aggregate: 是否进行聚合
    @return:
    '''
    arr = np.array([
        w2v_model.get_vector(s) for s in sentence
        if s in w2v_model.key_to_index
    ])
    if not aggregate:
        return arr
    if len(arr) > 0:
        # 第一种方法对一条样本中的词求平均
        if method == 'mean':
            return np.mean(np.array(arr), axis=0)
        # 第二种方法返回一条样本中的最大值
        elif method == 'max':
            return np.max(np.array(arr), axis=0)
        else:
            raise NotImplementedError
    else:
        return np.zeros(300)

def format_data(data, max_features, maxlen, tokenizer=None,shuffle=False):
    if shuffle:
        data = data.sample(frac=2).reset_index(drop=True)
    data['text']  = data['text'].apply(lambda x: " ".join(x))

    X = data['text']
    if not tokenizer:
        filters = "\"#$%&()*+./<=>@[\\]^_`{|}~\t\n"
        tokenizer = Tokenizer(num_words=max_features, filters=filters)
        tokenizer.fit_on_texts(list(X))
    X = tokenizer.texts_to_sequences(X)
    X = pad_sequences(X)

    return X, tokenizer

def formate_data(data, tfidf):
    Data = pd.concat([
        data[[
             'labelIndex', 'length', 'capitals', 'caps_vs_length',
            'num_exclamation_marks', 'num_question_marks', 'num_punctuation',
            'num_symbols', 'num_words', 'num_unique_words', 'words_vs_unique',
            'nouns', 'adjectives', 'verbs', 'nouns_vs_length',
            'adjectives_vs_length', 'verbs_vs_length', 'nouns_vs_words',
            'adjectives_vs_words', 'verbs_vs_words', 'count_words_title',
            'mean_word_len', 'punct_percent'
        ]], tfidf
    ] + [
        pd.DataFrame(
            data[i].tolist(),
            columns=[i + str(x) for x in range(data[i].iloc[0].shape[0])])
        for i in [
            'w2v_label_mean', 'w2v_label_max', 'w2v_mean', 'w2v_max',
            'w2v_win_2_mean', 'w2v_win_3_mean', 'w2v_win_4_mean',
            'w2v_win_2_max', 'w2v_win_3_max', 'w2v_win_4_max', 'res_embedding',
            'resnet_embedding', 'wide_embedding', 'lda', 'bert_embedding'
        ]
    ], axis=1).fillna(0.0)
    return Data
