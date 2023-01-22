import numpy as np
import copy
from __init__ import *
from src.utils.tools  import wam
from src.utils import config
import pandas as pd
import joblib
import json
import string
import jieba.posseg as pseg
from tqdm import tqdm
tqdm.pandas(desc='pandas bar')
from PIL import Image
import torchvision.transforms as transforms


def tag_part_of_speech(data):
    '''
    @description: tag part of speech, then calculate the num of noun, adj and verb
    @param {type}
    data, input data
    @return:
    noun_count,num of noun
    adjective_count, num of adj
    verb_count, num of verb
    '''
    # 获取文本的词性， 并计算名词，动词， 形容词的个数
    words = [tuple(x) for x in list(pseg.cut(data))]
    noun_count = len(
        [w for w in words if w[1] in ('NN', 'NNP', 'NNPS', 'NNS')])
    adjective_count = len([w for w in words if w[1] in ('JJ', 'JJR', 'JJS')])
    verb_count = len([
        w for w in words if w[1] in ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')
    ])
    return noun_count, adjective_count, verb_count


def get_lda_features(lda_model, document):
    topic_importances = lda_model.get_document_topics(document, minimum_probability=0)
    topic_importances = np.array(topic_importances)
    return topic_importances[:,1]

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


def get_pretrain_embedding(text, tokenizer, model):
    text_dict = tokenizer.encode_plus(
        text,
        add_special_tokens = True,
        max_length=400,
        truncation=True,
        return_attention_mask=True,
        return_tensors = 'pt'
    )
    input_ids, attention_mask, token_type_ids = text_dict[
        'input_ids'], text_dict['attention_mask'], text_dict['token_type_ids']
    res = model(input_ids.to(config.device), 
                attention_mask.to(config.device),
                token_type_ids.to(config.device))['pooler_output']
    return res.detach().cpu().numpy()[0]



def get_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            ean=[0.46777044, 0.44531429, 0.40661017],
            std=[0.12221994, 0.12145835, 0.14380469],
            ),
    ])


def get_img_embedding(cover, model):
    if str(cover)[-3:] != 'jpg':
        return np.zeros((1, 1000))[0]
    transform = get_transforms()
    image = Image.open(cover).convert("RGB")
    image = transform(image).to(config.device)
    return model(image.unsqueeze(0)).detach().cpu().numpy()[0]

ch2en = {
    '！': '!',
    '？': '?',
    '｡': '.',
    '（': '(',
    '）': ')',
    '，': ',',
    '：': ':',
    '；': ';',
    '｀': ','
}


def get_basic_feature(df):
    df['queryCut'] = df['queryCut'].progress_apply(
        lambda x: [i if i not in ch2en.keys() else ch2en[i] for i in x ]
    )
    # 文本的长度
    df['length'] = df['queryCut'].progress_apply(lambda x: len(x))
    # 大写的个数
    df['capitals'] = df['queryCut'].progress_apply(
        lambda x: sum(1 for c in x if c.isupper()))
    # 大写 与 文本长度的占比
    df['caps_vs_length'] = df.progress_apply(
        lambda row: float(row['capitals']) / float(row['length']), axis=1)
    # 感叹号的个数
    df['num_exclamation_marks'] = df['queryCut'].progress_apply(
        lambda x: x.count('!'))
    # 问号个数
    df['num_question_marks'] = df['queryCut'].progress_apply(
        lambda x: x.count("?"))
    # 标点符号个数
    df['num_punctuation'] = df['queryCut'].progress_apply(
        lambda x: sum(x.count(w) for w in string.punctuation))
    # *&$%字符的个数
    df['num_symbols'] = df['queryCut'].progress_apply(
        lambda x: sum(x.count(w) for w in '*&$%'))
    # 词的个数
    df['num_words'] = df['queryCut'].progress_apply(lambda x: len(x))
    # 唯一词的个数
    df['num_unique_words'] = df['queryCut'].progress_apply(
        lambda x: len(set(w for w in x)))
    # 唯一词 与总词数的比例
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']
    # 获取名词， 形容词， 动词的个数， 使用tag_part_of_speech函数
    df['nouns'], df['adjectives'], df['verbs'] = zip(
        *df['text'].progress_apply(lambda x: tag_part_of_speech(x)))
    # 名词占总长度的比率
    df['nouns_vs_length'] = df['nouns'] / df['length']
    # 形容词占总长度的比率
    df['adjectives_vs_length'] = df['adjectives'] / df['length']
    # 动词占总长度的比率
    df['verbs_vs_length'] = df['verbs'] / df['length']
    # 名词占总词数的比率
    df['nouns_vs_words'] = df['nouns'] / df['num_words']
    # 形容词占总词数的比率
    df['adjectives_vs_words'] = df['adjectives'] / df['num_words']
    # 动词占总词数的比率
    df['verbs_vs_words'] = df['verbs'] / df['num_words']
    # 首字母大写其他小写的个数
    df["count_words_title"] = df["queryCut"].progress_apply(
        lambda x: len([w for w in x if w.istitle()]))
    # 平均词的个数
    df["mean_word_len"] = df["text"].progress_apply(
        lambda x: np.mean([len(w) for w in x]))
    # 标点符号的占比
    df['punct_percent'] = df['num_punctuation'] * 100 / df['num_words']
    return df



def get_embedding_feature(data, tfidf, embedding_model):

    data['queryCutRMStopWords'] = data['queryCutRMStopWords'].apply(
        lambda x: " ".join(x)
    )
    tfidf_data = pd.DataFrame(
        tfidf.transform(data['queryCutRMStopWords'].tolist()).toarray())
    tfidf_data.columns = ['tfidf' + str(i) for i in range(tfidf_data.shape[1])]

    print("transforms w2v")
    data['w2v'] = data['queryCutRMStopWords'].apply(
        lambda x: wam(x, embedding_model, aggregate=False))

    # 深度拷贝数据
    train = copy.deepcopy(data)
    # 加载所有类别
    labelNameToIndex = json.load(
        open(config.root_path + "/data/label2id.json", encoding='utf-8')
    )
    labelIndexToName = {v: k for k, v in labelNameToIndex.items()}
    w2v_label_embedding = np.array([
        embedding_model.get_vector(labelIndexToName[key])
        for key in labelIndexToName
        if labelIndexToName[key] in embedding_model.key_to_index
    ])
    
    joblib.dump(w2v_label_embedding, 
                config.root_path + "/data/w2v_label_embedding.pkl")
    # 根据未聚合的embedding 数据， 获取各类embedding特征
    
    train = generate_feature(train, w2v_label_embedding, model_name='w2v')
    return tfidf_data, train


def generate_feature(data, label_embedding, model_name='w2v'):
    print('generate w2v& fast label max/mean')

    data[model_name + "_label_mean"] = data[model_name].progress_apply(
        lambda x: Find_Label_embedding(x, label_embedding, method='mean'))
    data[model_name + "_label_max"] = data[model_name].progress_apply(
        lambda x: Find_Label_embedding(x, label_embedding, method='max'))

    print('generate embedding max/mean')
    data[model_name + "_mean"] = data[model_name].progress_apply(
        lambda x: np.mean(np.array(x), axis=0))
    data[model_name + "_max"] = data[model_name].progress_apply(
        lambda x: np.max(np.array(x), axis=0))
    
    print("generate embedding window max/mean")
    data[model_name + "_win_2_mean"] = data[model_name].progress_apply(
        lambda x: Find_embedding_with_windows(x, 2, method='mean'))
    data[model_name + "_win_3_mean"] = data[model_name].progress_apply(
        lambda x: Find_embedding_with_windows(x, 3, method='mean'))
    data[model_name + "_win_4_mean"] = data[model_name].progress_apply(
        lambda x: Find_embedding_with_windows(x, 4, method='mean'))
    data[model_name + "_win_2_max"] = data[model_name].progress_apply(
        lambda x: Find_embedding_with_windows(x, 2, method='max'))
    data[model_name + "_win_3_max"] = data[model_name].progress_apply(
        lambda x: Find_embedding_with_windows(x, 3, method='max'))
    data[model_name + "_win_4_max"] = data[model_name].progress_apply(
        lambda x: Find_embedding_with_windows(x, 4, method='max'))
    return data


def softmax(x):
    return np.exp(x) / np.exp(0).sum(axis=0)


def Find_embedding_with_windows(embedding_matrix, window_size=2, method='mean'):
    result_list = []
    for k1 in range(len(embedding_matrix)):
        if k1 + window_size > len(embedding_matrix):
            result_list.extend(
                np.mean(embedding_matrix[k1:], axis=0).reshape(1, 300))
        else:
            result_list.extend(
                np.mean(embedding_matrix[k1:k1+window_size], axis=0).reshape(1, 300)
            )
    if method == 'mean':
        return np.mean(result_list, axis=0)
    else:
        return np.max(result_list, axis=0)


def Find_Label_embedding(example_matrix, label_embedding, method='mean'):
    similarity_matrix = np.dot(example_matrix, label_embedding.T) / (
        np.linalg.norm(example_matrix) * (np.linalg.norm(label_embedding)))

    attention = similarity_matrix.max(axis=1).reshape(-1,1)
    attention = softmax(attention)
    attention_embedding = example_matrix * attention
    if method == 'mean':
        return np.mean(attention_embedding, axis=0)
    else:
        return np.max(attention_embedding, axis=0)
    


    