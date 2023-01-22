import os

import lightgbm as lgb
import numpy as np
import torchvision
import json
import pandas as pd
from src.utils import config
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from sklearn.ensemble import RandomForestClassifier
import joblib
from src.data.mlData import MLData
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from transformers import BertModel, BertTokenizer
from src.utils.tools import formate_data, get_score,create_logger
from src.utils.feature import (get_embedding_feature, get_basic_feature,
                                 get_img_embedding, get_lda_features,get_pretrain_embedding)

from __init__ import *
logger = create_logger(config.log_dir + "model.log")

class Models(object):
    def __init__(self,
                model_path=None,
                feature_engineer=None,
                train_mode=True):
        self.model_name = None
        self.res_model = torchvision.models.resnet152(
            pretrained=True
        )
        self.res_model = self.res_model.to(config.device)
        self.resnext_model = torchvision.models.resnext101_32x8d(
            pretrained=True
        )
        self.wide_model = torchvision.models.wide_resnet101_2(pretrained=True)
        self.wide_model = self.wide_model.to(config.device)
        # 加载bert模型
        self.bert_tonkenizer = BertTokenizer.from_pretrained(config.root_path + "/model/bert")

        self.bert = BertModel.from_pretrained(config.root_path + "/model/bert")
        self.bert = self.bert.to(config.device)

        # 初始化MLdataset类
        self.ml_data = MLData(debug_mode=True, train_mode=train_mode)

        # 加载已经训练好的模型
        if train_mode:
            self.model = lgb.LGBMClassifier(objective='multiclass',
                                            n_jobs=10,
                                            num_class=33,
                                            num_leaves=30,
                                            reg_alpha=10,
                                            reg_lambda=200,
                                            max_depth=3,
                                            learning_rate=0.05,
                                            n_estimators=2000,
                                            bagging_freq=1,
                                            bagging_fraction=0.9,
                                            feature_fraction=0.8,
                                            seed=1440)
        else:
            self.load(model_path)
            labelNameToIndex = json.load(
                open(config.root_path + "/data/label2id.json",
                encoding='utf-8'))
            self.ix2label = {v: k for k , v in labelNameToIndex.items()}
    
    def feature_engineer(self):
        
        logger.info("generate embedding feature ")

        train_tfidf, train = get_embedding_feature(self.ml_data.train, 
                                            self.ml_data.em.tfidf,
                                            self.ml_data.em.w2v)
        test_tfidf, test = get_embedding_feature(self.ml_data.dev, 
                                            self.ml_data.em.tfidf,
                                            self.ml_data.em.w2v)
        
        logger.info("generate atuoencoder feature")
        
        logger.info("generate basic feature")
        # 获得nlp 基本特征
        train = get_basic_feature(train)
        test = get_basic_feature(test)

        logger.info("generate model feature")
        # 加载图书封面
        cover = os.listdir(config.root_path + "/data/book_cover/")
        # 根据title匹配图书封面
        train['cover'] = train['title'].progress_apply(
            lambda x: config.root_path + "/data/book_cover/" + x + '.jpg'
            if x + ".jpg" in cover else "")
        test['cover'] = test['title'].progress_apply(
            lambda x: config.root_path + "/data/book_cover/" + x + '.jpg'
            if x + ".jpg" in cover else "")

        # 根据封面获取封面的embedding
        train['res_embedding'] = train['cover'].progress_apply(
            lambda x: get_img_embedding(x, self.res_model))
        test['res_embedding'] = test['cover'].progress_apply(
            lambda x: get_img_embedding(x, self.res_model))
        
        train['resnet_embedding'] = train['cover'].progress_apply(
            lambda x: get_img_embedding(x, self.resnext_model))
        test['resnet_embedding'] = test['cover'].progress_apply(
            lambda x: get_img_embedding(x, self.resnext_model))
        
        train['wide_embedding'] = train['cover'].progress_apply(
            lambda x: get_img_embedding(x, self.wide_model))
        test['wide_embedding'] = test['cover'].progress_apply(
            lambda x: get_img_embedding(x, self.wide_model))

        logger.info("generate bert feature")
        train['bert_embedding'] = train['text'].progress_apply(
            lambda x: get_pretrain_embedding(x, self.bert_tonkenizer, self.bert))
        test['bert_embedding'] = test['text'].progress_apply(
            lambda x: get_pretrain_embedding(x, self.bert_tonkenizer, self.bert))
        
        logger.info("generate lda feature")
        train['bow'] = train['queryCutRMStopWords'].apply(
            lambda x: self.ml_data.em.lda.id2word.doc2bow(x.split()))
        test['bow'] = test['queryCutRMStopWords'].apply(
            lambda x: self.ml_data.em.lda.id2word.doc2bow(x.split()))
        # the bag  of word 基础上
       
        train['lda'] = list(
            map(lambda doc: get_lda_features(self.ml_data.em.lda, doc),
                train['bow']))
        test['lda'] = list(
            map(lambda doc: get_lda_features(self.ml_data.em.lda, doc),
                test['bow']))
        
        logger.info('formate data')
        # 将所有特征拼接
        train = formate_data(train, train_tfidf)
        test = formate_data(test, test_tfidf)
        # 生成训练，测试的数据

        cols = [x for x in train.columns if str(x) != 'labelIndex']
        X_train, X_test = train[cols], test[cols]
        train['labelIndex'] = train['labelIndex'].astype(int)
        test['labelIndex'] = test['labelIndex'].astype(int)

        y_train = train['labelIndex']
        y_test = test['labelIndex']
        return X_train, X_test, y_train, y_test

    def process(self, title, desc):
        df = pd.DataFrame([[title, desc]], columns=['title', 'desc'])
        df['text'] = df['title'] + df['desc']

        df["queryCut"] = df["text"].apply(query_cut)
        df["queryCutRMStopWords"] = df["queryCut"].apply(
            lambda x:
            [word for word in x if word not in self.ml_data.em.stopWords])
        
        df_tfidf, df = get_embedding_feature(df, self.ml_data.em.tfidf,
                                             self.ml_data.em.w2v)
        print("generate basic feature ")
        df = get_basic_feature(df)

        print("generate modal feature ")
        df['cover'] = ''
        df['res_embedding'] = df.cover.progress_apply(
            lambda x: get_img_embedding(x, self.res_model))

        df['resnext_embedding'] = df.cover.progress_apply(
            lambda x: get_img_embedding(x, self.resnext_model))

        df['wide_embedding'] = df.cover.progress_apply(
            lambda x: get_img_embedding(x, self.wide_model))

        print("generate bert feature ")
        df['bert_embedding'] = df.text.progress_apply(
            lambda x: get_pretrain_embedding(x, self.bert_tonkenizer, self.bert
                                             ))

        print("generate lda feature ")
        df['bow'] = df['queryCutRMStopWords'].apply(
            lambda x: self.ml_data.em.lda.id2word.doc2bow(x))
        df['lda'] = list(
            map(lambda doc: get_lda_features(self.ml_data.em.lda, doc),
                df.bow))

        print("formate data")
        df['labelIndex'] = 1
        df = formate_data(df, df_tfidf)
        cols = [x for x in df.columns if str(x) not in ['labelIndex']]
        X_train = df[cols]
        return X_train

    def unbalance_helper(self, 
                        imbalance_method='under_sampling',
                        search_method='grid'):
        logger.info("get all freature")
        self.X_train, self.X_test, self.y_train, self.y_test = self.feature_engineer()
        if imbalance_method == 'over_sampling':
            logger.info("Use SMOTE deal with unbalance data")
            self.X_train, self.y_train = SMOTE().fit_resample(
                self.X_train, self.y_train)
            self.X_test, self.y_test = SMOTE().fit_resample(
                self.X_test, self.y_test)
            self.model_name = 'lgb_over_sampling'
        elif imbalance_method == 'under_sampling':
            logger.info("Use ClusterCentroids deal with unbalance data ")
            self.X_train, self.y_train = ClusterCentroids().fit_resample(
                self.X_train, self.y_train)
            self.X_test, self.y_test = ClusterCentroids().fit_resample(
                self.X_test, self.y_test)
            self.model_name = 'lgb_under_sampling'
        elif imbalance_method == 'ensemble':
            self.model = BalancedBaggingClassifier(
                base_estimator=DecisionTreeClassifier(),
                sampling_strategy='auto',
                replacement=False,
                random_state=0)
            self.model_name = 'ensemble'

        if imbalance_method != 'ensemble':
            # param = self.param_search(search_method=search_method)
            # param['params']['num_leaves'] = int(param['params']['num_leaves'])
            # param['params']['max_depth'] = int(param['params']['max_depth'])
            param = {}
            param['params'] = {}
            param['params']['num_leaves'] = 3
            param['params']['max_depth'] = 5
            self.model = self.model.set_params(**param['params'])
        logger.info('fit model ')
        # 训练， 并输出模型的结果
        self.model.fit(self.X_train, self.y_train)
        Test_predict_label = self.model.predict(self.X_test)
        Train_predict_label = self.model.predict(self.X_train)
        per, acc, recall, f1 = get_score(self.y_train, self.y_test,
                                         Train_predict_label,
                                         Test_predict_label)
        # 输出训练集的精确率
        logger.info('Train accuracy %s' % per)
        # 输出测试集的准确率
        logger.info('test accuracy %s' % acc)
        # 输出recall
        logger.info('test recall %s' % recall)
        # 输出F1-score
        logger.info('test F1_score %s' % f1)
        self.save(self.model_name)

    
    def predict(self, title, desc):
        inputs = self.process(title, desc)
        label = self.ix2label[self.model.predict(inputs)[0]]
        proba = np.max(self.model.predict_proba(inputs))
        return label, proba

    def save(self, model_name):
        joblib.dump(self.model, config.root_path + "/model/ml_model/" + model_name)

    def load(self, path):
        self.model = joblib.load(path)


    
        



    

        
        



        


        


