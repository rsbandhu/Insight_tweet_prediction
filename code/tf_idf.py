import numpy as np
import os
import collections
import bz2
import pickle
import argparse

from scipy.sparse import csr_matrix

class tf_idf(object):

    def __init__(self, term_document_count):
        self.idf_dict = {}
        print("number of items in tf_df dict: {}".format(len(self.idf_dict)))
        self.term_doc_count_to_idf(term_document_count) #generates idf_dict
        print(len(self.idf_dict))
        self.vocab_keys = [key for key in self.idf_dict.keys()]

    def raw_count(self, tokens_list):
        '''
        returns a dict of raw counts of each iten in the vocab for a line
        '''
        term_count_dict = {key:0 for key in self.vocab_keys}
        
        for item in tokens_list:
            term_count_dict[item] += 1
        
        return term_count_dict

    def doc_frequency(self, tokens_list, doc_freq_dict):
        '''
        updates the term frequenct count given a batch of list of tokens
        '''
        item_present = collections.defaultdict(int)

        for token in tokens_list:
            if item_present[token] == 1:
                continue
            else:
                doc_freq_dict[token] += 1
                item_present[token] += 1
        
        return doc_freq_dict

    def create_batch(self, tokens_batch):
        '''
        generates a sparse matrix for a batch of tokens, consisting of tf-idf values of tokens
        input:
        tokens_batch: batch of list. each list consists of tokens from a single tweet
        returns:
        tf_idf_batch: tf_idf count matrix: of each tokens in tweets in the batch
        sparse matrix (csr_matrix format)
        '''
        tf_idf_batch = []

        for i in range(len(tokens_batch)):
            tokens_list = tokens_batch[i]
            tf_dict = self.raw_count(tokens_list)
            tf_idf = [tf_dict[k] * self.idf_dict[k] for k in self.vocab_keys]
            tf_idf_batch.append(tf_idf)

        return tf_idf_batch
        #return csr_matrix(tf_idf_batch)

    def term_doc_count_to_idf(self, term_document_count):
        '''
        generates idf dict from term trequency in document
        '''
        N = term_document_count['101']
    
        for key in term_document_count.keys():
            self.idf_dict[key] = np.log(N/term_document_count[key])


def main():

   
    #train_file = "train_first_1000.tsv"
    #train_file = "val.tsv"

    if args.mode != 'tf_idf':
        train_file = "training.tsv"
        data_dir = "../data"
        file_path = os.path.join(data_dir, train_file)
        fhandle = open(file_path, 'r')


        TR = tweetrecords.TweetRecords(file_path, 1)

        output_file = 'document_freq_count_train.pickle' 
        outf_handle = bz2.BZ2File(os.path.join(data_dir, output_file), 'w')


        doc_freq_count = collections.defaultdict(int)

        TF_IDF = tf_idf()

        line = TR.file_handle.readline()
        line_count = 1
        while line:
            tokens = TR.parse_line(line)[0].split()
            #print(tokens[0])
            doc_freq_count = TF_IDF.doc_frequency(tokens, doc_freq_count)

            line = TR.file_handle.readline()
            if (line_count % 1000000 == 0):
                print(line_count)
            line_count += 1

        TR.close_file()
        pickle.dump(doc_freq_count, outf_handle)


if __name__ == '__main__':
    main()

    



