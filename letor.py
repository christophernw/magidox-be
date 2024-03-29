import lightgbm as lgb
import numpy as np
import re
import random

#from mpstemmer import MPStemmer
#from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
# from mpstemmer import MPStemmer
# from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


from scipy.spatial.distance import cosine
from gensim.corpora import Dictionary
from gensim.models import LsiModel
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize

from string import punctuation
from spellchecker import SpellChecker

punctuation = list(punctuation)

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

class Cleaner:
  # stemmer = PorterStemmer()
  stemmer = SnowballStemmer("english")
  stop_words = stopwords.words('english')

  @staticmethod
  def clean_and_tokenize(uncleaned_sentence):
    tokenized_words = word_tokenize(uncleaned_sentence)
    stemmed = [Cleaner.stemmer.stem(word) for word in tokenized_words]
    cleaned_tokens = []
    for token in stemmed:
      if (token not in Cleaner.stop_words) and (token not in punctuation):
        cleaned_tokens.append(token)
    return cleaned_tokens

  @staticmethod
  def tokenize(uncleaned_sentence):
    tokenized_words = word_tokenize(uncleaned_sentence)
    cleaned_tokens = []
    for token in tokenized_words:
      if (token not in punctuation):
        cleaned_tokens.append(token)
    return cleaned_tokens
  
'''
Pada TP 3, saya kemas code utama nya nya ke dalam class LambdaMart
'''
class LambdaMart:
    NUM_LATENT_TOPICS = 200
    
    NUM_NEGATIVES = 1

    def __init__(self):
        self.documents = {}
        
        self.queries = {}
        self.val_queries = {}

        self.q_docs_rel = {}
        self.val_q_docs_rel = {}

        self.group_qid_count = []
        self.val_group_qid_count = []

        self.dataset = []
        self.val_dataset = []


        self.dictionary = Dictionary()
        
        # keperluan untuk "training the ranker"
        self.ranker = lgb.LGBMRanker(
            objective="lambdarank",
            boosting_type = "gbdt",
            n_estimators = 600,
            importance_type = "gain",
            metric = "ndcg",
            num_leaves = 60,
            learning_rate = 0.01,
            max_depth = -1,
        )
        # training
        self.load_documents('qrels-folder/train_docs.txt')
        self.load_queries('qrels-folder/train_queries.txt')
        self.load_qrels('qrels-folder/train_qrels.txt')
        self.construct_dataset()

        # validation
        self.load_val_queries('qrels-folder/val_queries.txt')
        self.load_val_qrels('qrels-folder/val_qrels.txt')
        self.construct_val_dataset()

        # model
        self.build_lsi_model()
        self.fit_dataset()
    
    def _preprocess_line(self, line):
        # # lakukan stem terlebih dahulu
        # stemmed_line = self.stemmer.stem_kalimat(line)
        
        # # hapus stop word
        # preprocessed_line = self.stop_word_remover.remove(stemmed_line)
        # return re.findall(r'\w+', preprocessed_line)
        return Cleaner.clean_and_tokenize(line)

    def load_documents(self, train_docs_path):
        with open(train_docs_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

            for line in lines:
                doc_id, content = line.strip().split(' ', 1)
                # lakukan preposes pada content doc dulu
                self.documents[doc_id] = self._preprocess_line(content)

    
    def load_queries(self, train_queries_path):
        with open(train_queries_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

            for line in lines:
                q_id, content = line.strip().split(' ', 1)
                # lakukan preposes pada content query dulu
                self.queries[q_id] = self._preprocess_line(content)
        
    
    def load_val_queries(self, val_queries_path):
        with open(val_queries_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

            for line in lines:
                q_id, content = line.strip().split(' ', 1)
                self.val_queries[q_id] = self._preprocess_line(content)
        

    def load_qrels(self, train_qrel_path):
        with open(train_qrel_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

            for line in lines:
                q_id, doc_id, rel = line.strip().split()
                if (q_id in self.queries) and (doc_id in self.documents):
                    if q_id not in self.q_docs_rel:
                        self.q_docs_rel[q_id] = []
                    self.q_docs_rel[q_id].append((doc_id, int(rel)))

    
    def load_val_qrels(self, val_qrels_path):
        with open(val_qrels_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

            for line in lines:
                q_id, doc_id, rel = line.strip().split()
                if (q_id in self.val_queries) and (doc_id in self.documents):
                    if q_id not in self.val_q_docs_rel:
                        self.val_q_docs_rel[q_id] = []
                    self.val_q_docs_rel[q_id].append((doc_id, int(rel)))
        
    def construct_dataset(self):
        for q_id in self.q_docs_rel:
            docs_rels = self.q_docs_rel[q_id]
            self.group_qid_count.append(len(docs_rels) + LambdaMart.NUM_NEGATIVES)

            for doc_id, rel in docs_rels:
                self.dataset.append((self.queries[q_id], self.documents[doc_id], rel))

            for _ in range(LambdaMart.NUM_NEGATIVES):
                self.dataset.append((self.queries[q_id], random.choice(list(self.documents.values())), 0))

        assert sum(self.group_qid_count) == len(self.dataset)
    
    def construct_val_dataset(self):
        for q_id in self.val_q_docs_rel:
            docs_rels = self.val_q_docs_rel[q_id]

            self.val_group_qid_count.append(len(docs_rels) + LambdaMart.NUM_NEGATIVES)
            for doc_id, rel in docs_rels:
                self.val_dataset.append((self.val_queries[q_id], self.documents[doc_id], rel))
            
            for _ in range(LambdaMart.NUM_NEGATIVES):
                self.val_dataset.append((self.val_queries[q_id], random.choice(list(self.documents.values())), 0))
        
        assert sum(self.val_group_qid_count) == len(self.val_dataset)
    
    def build_lsi_model(self):
        bow_corpus = [self.dictionary.doc2bow(doc, allow_update=True) for doc in self.documents.values()]
        self.model = LsiModel(bow_corpus, num_topics=LambdaMart.NUM_LATENT_TOPICS)

    def _vector_rep(self, text):
        rep = [topic_value for (_, topic_value) in self.model[self.dictionary.doc2bow(text)]]
        return rep if len(rep) == LambdaMart.NUM_LATENT_TOPICS else [0.] * LambdaMart.NUM_LATENT_TOPICS

    def features(self, query, doc):

        v_q = self._vector_rep(query)
        v_d = self._vector_rep(doc)
        q = set(query)
        d = set(doc)
        cosine_dist = cosine(v_q, v_d)
        jaccard = len(q & d) / len(q | d)

        return v_q + v_d + [jaccard] + [cosine_dist]
    
    def fit_dataset(self):
        # kita ubah dataset menjadi X dan Y
        # dimana X adalah representasi gabungan query+document,
        # dan Y adalah label relevance untuk query dan document tersebut.
        X = []
        Y = []

        for (query, doc, rel) in self.dataset:
            X.append(self.features(query, doc))
            Y.append(rel)
        
        # ubah X dan Y ke format numpy array
        X = np.array(X)
        Y = np.array(Y)

        # validation
        X_val = []
        Y_val = []
        for (query, doc, rel) in self.val_dataset:
            X_val.append(self.features(query, doc))
            Y_val.append(rel)
        
        X_val = np.array(X_val)
        Y_val = np.array(Y_val)

        self.ranker.fit(X, Y, group=self.group_qid_count, eval_set=[(X_val, Y_val)], eval_group=[self.val_group_qid_count], eval_metric='ndcg')
        
        print(self.ranker.best_score_)
    
    def predict(self, X):
        return self.ranker.predict(X)
    
    def evaluate_letor(self, query, doc_path):
        if not doc_path:
            return []

        X = []
        for doc in doc_path:
            with open(doc, 'r', encoding='utf-8') as f:
                X.append(self.features(self._preprocess_line(query), self._preprocess_line(f.readline())))

        X = np.array(X)
        scores = self.predict(X)

        # Ranking pada SERP 
        did_scores = [x for x in zip(scores, doc_path)]
        sorted_did_scores = sorted(did_scores, key = lambda tup: tup[0], reverse = True)

        return sorted_did_scores

if __name__ == '__main__':
    # untuk coba run code ini
    lambda_mart = LambdaMart()
