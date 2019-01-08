#!/usr/bin/env python

import nltk, zipfile, argparse

###############################################################################
## Utility Functions ##########################################################
###############################################################################
# This method takes the path to a zip archive.
# It first creates a ZipFile object.
# Using a list comprehension it creates a list where each element contains
# the raw text of the fable file.
# We iterate over each named file in the archive:
#     for fn in zip_archive.namelist()
# For each file that ends with '.txt' we open the file in read only
# mode:
#     zip_archive.open(fn, 'rU')
# Finally, we read the raw contents of the file:
#     zip_archive.open(fn, 'rU').read()
def unzip_corpus(input_file):
    zip_archive = zipfile.ZipFile(input_file)
    try:
        contents = [zip_archive.open(fn, 'rU').read().decode('utf-8')
                for fn in zip_archive.namelist() if fn.endswith(".txt")]
    except ValueError as e:
        contents = [zip_archive.open(fn, 'r').read().decode('utf-8')
                for fn in zip_archive.namelist() if fn.endswith(".txt")]
    return contents

###############################################################################
## Stub Functions #############################################################
###############################################################################
def process_corpus(corpus_name):
    input_file = corpus_name + ".zip"
    corpus_contents = unzip_corpus(input_file)

    # Your code goes here
    #print the name of the corpus
    print(corpus_name)
    #divide each part of the corpus into sentences
    sentences = [nltk.sent_tokenize(part) for part in corpus_contents]
    #divide each sentence into words
    words = [[nltk.word_tokenize(sent) for sent in part] for part in sentences]
    print(words[1])
    #print the total number of words in the corpus
    #print(total_words)
    #print the vocabulary size of the corpus
    #print(vocabulary)
    #print the most frequent part of the speech (with frequency)
    #print most frequent words for the different POS(NN, VBD)
    pass

###############################################################################
## Program Entry Point ########################################################
###############################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assignment 1')
    parser.add_argument('--corpus', required=True, dest="corpus", metavar='NAME',
                        help='Which corpus to process {fables, blogs}')

    args = parser.parse_args()
    
    corpus_name = args.corpus
    
    if corpus_name == "fables" or "blogs":
        process_corpus(corpus_name)
    else:
        print("Unknown corpus name: {0}".format(corpus_name))
        