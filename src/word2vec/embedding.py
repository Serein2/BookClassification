import pandas as pd


from __init__ import *

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import gensim
from gensim import models
from gensim.models  import LdaMulticore
from gensim.models.ldamodel import LdaModel
from src.utils.config import root_path
from src.utils.tools import create_logger, query_cut
logger = create_logger(root_path + '/logs/embedding.log')


class SingletonMetaclass(type):
    '''
    @description: singleton
    '''
    def __init__(self, *args, **kwargs):
        self.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.__instance is None:
            self.__instance = super(SingletonMetaclass,
                                    self).__call__(*args, **kwargs)
            return self.__instance
        else:
            return self.__instance

class Embedding(metaclass=SingletonMetaclass):
    def __init__(self):
        self.stopWords = open(root_path + "/data/stopwords.txt", encoding='utf-8').readlines()

    def load_data(self):
        logger.info("load data")
        self.data = pd.concat([
            pd.read_csv(root_path + "/data/train_clean.csv", sep='\t'),
            pd.read_csv(root_path + "/data/dev_clean.csv", sep='\t'),
            pd.read_csv(root_path + "/data/test_clean.csv", sep='\t'),
        ])
        self.data['text'] = self.data['text'].apply(lambda x: x.split()) + self.data['label'].apply(lambda x: [x])
    
    def trainer(self):
        logger.info("train tfidf")
        count_vect = TfidfVectorizer(stop_words=self.stopWords, 
                                    max_df=0.4,
                                    min_df=0.001,
                                    ngram_range=(1, 2) )
        self.tfidf = count_vect.fit(self.data['text'].apply(lambda x: " ".join(x)))
        logger.info("train word2vec")

        self.w2v = models.Word2Vec(min_count=2,
                                    window=5,
                                    vector_size=300,
                                    alpha=0.03,
                                    min_alpha=0.0007,
                                    negative=15,
                                    workers=4,
                                    epochs=30,
                                    max_vocab_size=50000)
        self.w2v.build_vocab(self.data['text'])
        self.w2v.train(self.data['text'], 
                        total_examples=self.w2v.corpus_count,
                        epochs=15,
                        report_delay=1)
        logger.info("train fast")
        # 训练fast的向量
        self.fast = models.FastText(
                self.data['text'],
                vector_size=300,
                window=3,
                alpha=0.03,
                min_count=2,
                epochs=30,
                max_n=3,
                word_ngrams=1,
                max_vocab_size=50000
        )
        logger.info("train lda")
        self.id2word = gensim.corpora.Dictionary(self.data.text)
        corpus = [self.id2word.doc2bow(text) for text in self.data.text]
        self.LDAmodel = LdaMulticore(corpus=corpus, id2word=self.id2word, 
                                        num_topics=30,
                                        workers=4,
                                        chunksize=4000,
                                        passes=7,
                                        alpha='asymmetric')
    
    def saver(self):
        logger.info("save tfidf model")
        joblib.dump(self.tfidf, root_path + "/model/embedding/tfidf")

        logger.info("save w2v model")
        self.w2v.wv.save_word2vec_format(root_path + "/model/embedding/w2v.bin", binary=True)

        logger.info("save fast model")
        self.fast.wv.save_word2vec_format(root_path + "/model/embedding/fast.bin", binary=True)

        logger.info("save ladmodel")
        self.LDAmodel.save(root_path + "/model/embedding/lda")
    
    def load(self):
        logger.info("load tfidf model")
        self.tfidf = joblib.load(root_path + "/model/embedding/tfidf")

        logger.info("load w2v model")
        self.w2v = models.KeyedVectors.load_word2vec_format(
            root_path + "/model/embedding/w2v.bin", binary=True
        )

        logger.info("load fast model")
        self.fast = models.KeyedVectors.load_word2vec_format(
            root_path + "/model/embedding/fast.bin", binary=True
        )
        
        logger.info("load lda model")
        self.lda = LdaModel.load(root_path + "/model/embedding/lda")

if __name__ == '__main__':
    em = Embedding()
    em.load_data()
    em.trainer()
    em.saver()


