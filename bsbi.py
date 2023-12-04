import glob
import os
import pickle
import contextlib
import heapq
import math
import re

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, merge_and_sort_posts_and_tfs
from compression import VBEPostings
from tqdm import tqdm

# from mpstemmer import MPStemmer
# from nlp_id.stopword import StopWord
# from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from operator import itemgetter

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

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """

    def __init__(self, data_dir, output_dir, postings_encoding, index_name="main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding
        self.spellchecker = SpellChecker()

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'tokenized_terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)
        with open(os.path.join(self.output_dir, 'spellchecker.dict'), 'wb') as f:
            pickle.dump(self.spellchecker, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, 'tokenized_terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'spellchecker.dict'), 'rb') as f:
            self.spellchecker = pickle.load(f)

    def init_spellchecker(self):
        for filename in tqdm(
            glob.glob(f"{self.data_dir}/**/*.txt", recursive=True)):
            print(filename, "INI FILE NAME COBA PRINT")
            with open(filename, "r", encoding='utf-8') as buffer:
                sentence = buffer.read()
                sentence = Cleaner.tokenize(sentence)
                # print(sentence)
                self.spellchecker.word_frequency.load_words(sentence)

    def pre_processing_text(self, content):
        """
        Melakukan preprocessing pada text, yakni stemming dan removing stopwords
        """
        # https://github.com/ariaghora/mpstemmer/tree/master/mpstemmer

        # stemmer = MPStemmer()
        # stemmed = stemmer.stem(content)
        # remover = StopWordRemoverFactory().create_stop_word_remover()
        # return remover.remove(stemmed)

        return Cleaner.clean_and_tokenize(content)

    def parsing_block(self, block_path):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk stemming bahasa Indonesia, seperti
        MpStemmer: https://github.com/ariaghora/mpstemmer 
        Jangan gunakan PySastrawi untuk stemming karena kode yang tidak efisien dan lambat.

        JANGAN LUPA BUANG STOPWORDS! Kalian dapat menggunakan PySastrawi 
        untuk menghapus stopword atau menggunakan sumber lain seperti:
        - Satya (https://github.com/datascienceid/stopwords-bahasa-indonesia)
        - Tala (https://github.com/masdevid/ID-Stopwords)

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_path : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parsing_block(...).
        """
        # TODO

        # stemmer = MPStemmer()
        # stop_word_remover = StopWord()

        term_doc_list = []

        for entry in os.scandir(os.path.join(self.data_dir, block_path)):
            if entry.is_file():
                with open(entry.path, 'r', encoding='utf-8') as f:
                    for line in f:    
                        # line_without_stopwords = stop_word_remover.remove_stopword(line)
                        # terms = re.findall(r'\w+', line_without_stopwords)
                        terms = self.pre_processing_text(line)

                        for term in terms:
                            # stemmed_term = stemmer.stem(term.lower())
                            term_id = self.term_id_map[term]
                            doc_id = self.doc_id_map[os.path.join(self.data_dir, block_path, entry.name)]

                            term_doc_list.append((term_id, doc_id))

        return term_doc_list

    def write_to_index(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-maintain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan strategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        # TODO
        term_dict = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = {}
            term_dict[term_id][doc_id] = term_dict[term_id].get(doc_id, 0) + 1

        for term_id in sorted(term_dict.keys()):
            original_dict = term_dict[term_id]
            sorted_dict = dict(sorted(original_dict.items()))
            index.append(term_id, list(sorted_dict.keys()), list(sorted_dict.values()))

    def merge_index(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi merge_and_sort_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key=lambda x: x[0])
        curr, postings, tf_list = next(merged_iter)  # first item
        for t, postings_, tf_list_ in merged_iter:  # from the second item
            if t == curr:
                zip_p_tf = merge_and_sort_posts_and_tfs(list(zip(postings, tf_list)),
                                                        list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def retrieve_tfidf(self, query, k=10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        # TODO

        # stemmer = MPStemmer()
        # stop_word_remover = StopWord()
        
        # query_without_stopwords = stop_word_remover.remove_stopword(query)
        # terms = re.findall(r'\w+', query_without_stopwords)

        # query_terms = []
        
        # for term in terms:
        #     stemmed_term = stemmer.stem(term.lower())
        #     query_terms.append(stemmed_term)
        
        query_terms = self.pre_processing_text(query)

        doc_score = {}

        with InvertedIndexReader(self.index_name, self.postings_encoding, self.output_dir) as reader:
            N = len(reader.doc_length)

            for term in query_terms:
                if term not in self.term_id_map:
                    continue

                term_id = self.term_id_map[term]
                posting_list, tf_list = reader.get_postings_list(term_id)
                
                df = len(posting_list)
                idf = math.log(N/df, 10)

                for doc_id, tf in zip(posting_list, tf_list):
                    wtd = 1 + math.log(tf, 10)
                    score = wtd * idf

                    doc_score[doc_id] = doc_score.get(doc_id, 0) + score

        return sorted([
            (doc_score[doc_id], self.doc_id_map[doc_id])
            for doc_id in doc_score
        ], reverse=True)[:k] 

    def retrieve_bm25(self, query, k=10, k1=1.2, b=0.75):
        """
        Melakukan Ranked Retrieval dengan skema scoring BM25 dan framework TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        """
        # TODO

        # print("bm_25")
        # stemmer = MPStemmer()
        # stop_word_remover = StopWord()
        
        # query_without_stopwords = stop_word_remover.remove_stopword(query)
        # terms = re.findall(r'\w+', query_without_stopwords)

        # query_terms = []
        # for term in terms:
        #     stemmed_term = stemmer.stem(term.lower())
        #     query_terms.append(stemmed_term)
        query_terms = self.pre_processing_text(query)

        doc_score = {}

        with InvertedIndexReader(self.index_name, self.postings_encoding, self.output_dir) as reader:
            N = len(reader.doc_length)
            avdl = sum(reader.doc_length.values()) / len(reader.doc_length)

            # print(len(query_terms), "ini query terms")
            for term in query_terms:
                if term not in self.term_id_map:
                    continue

                term_id = self.term_id_map[term]
                posting_list, tf_list = reader.get_postings_list(term_id)
                
                df = len(posting_list)
                # print(df)

                for doc_id, tf in zip(posting_list, tf_list):
                    dl = reader.doc_length[doc_id]
                    score = math.log(N/df, 10) * ((k1+1) * tf) / (k1 * ((1-b) + b*dl/avdl) + tf)

                    doc_score[doc_id] = doc_score.get(doc_id, 0) + score

        hasil = sorted([
            (doc_score[doc_id], self.doc_id_map[doc_id])
            for doc_id in doc_score
        ], reverse=True)[:k]

        print(len(hasil), " ini hasilll")
        return hasil
    
    def spellcheck(self, queries):
        tokenized_words = Cleaner.tokenize(queries.lower())
        spellchecked = []
        changed = False
        for q in tokenized_words:
            curq = self.spellchecker.correction(q)
             # print(q, curq)
            spellchecked.append(curq if curq != None else q)
            if (spellchecked[-1] != q.lower()):
                changed = True
        return ' '.join(spellchecked), changed
    
    def do_indexing(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parsing_block
        untuk parsing dokumen dan memanggil write_to_index yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parsing_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory=self.output_dir) as index:
                self.write_to_index(td_pairs, index)
                td_pairs = None

        self.init_spellchecker()
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                           for index_id in self.intermediate_indices]
                self.merge_index(indices, merged_index)


if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir='collections',
                              postings_encoding=VBEPostings,
                              output_dir='index')
    BSBI_instance.do_indexing()  # memulai indexing!
