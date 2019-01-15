#!/usr/bin/env python

import nltk, zipfile, argparse, sys

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

    # print the name of the corpus
    print("Corpus name:", corpus_name)

    """
    Tokenization
    """

    # divide each part of the corpus into sentences
    sentences = [nltk.sent_tokenize(part) for part in corpus_contents]

    # divide each sentence into words
    words = [nltk.word_tokenize(sent) for part in sentences for sent in part]

    # get a flat list of words
    flat_words = [word for sentence in words for word in sentence]

    # print the total number of words in the corpus
    print("Total words in the corpus:", len(flat_words))


    """
    Part-of-Speech
    """

    # tag the words of the corpus for each sentence
    tagged_senteces = [[nltk.pos_tag(nltk.word_tokenize(sent)) for sent in part] for part in sentences]

    # print each sentenced with tagged words in a file
    with open(corpus_name + '-pos.txt', 'w') as f:
        for part in tagged_senteces:
            for sent in part:
                for tuple in sent:
                    f.write(tuple[0] + '/' + tuple[1] + ' ')
                f.write('\n')
            f.write('\n')

    # get the frequency of the parts-of-speech
    tagged_words = [tuple for part in tagged_senteces for sent in part for tuple in sent]
    tag_fd = nltk.FreqDist(tag for (word, tag) in tagged_words)

    # print the most frequent part of the speech
    print("The most frequent part-of-speech is", tag_fd.most_common()[0][0], "with frequency", tag_fd.most_common()[0][1])

    """
    Frequency
    """

    # lowercase all the words in the corpus
    l_flat_words = [word.lower() for word in flat_words]

    # print the size of the vocabulary
    print("Vocabulary size of the corpus:", len(set(l_flat_words)))

    # get the frequency distribution for the words in the corpus
    freq_dist = nltk.FreqDist(l_flat_words)
    # print the words in descent order by frequency
    with open(corpus_name + '-word-freq.txt', 'w') as f:
        f.write(", ".join("(%s,%s)" % tup for tup in freq_dist.most_common()))

    # get a flat list of the tagged words
    flat_tagged_words = nltk.pos_tag(l_flat_words)

    # get the reversed tuples
    rev_tagged_words = [(b,a) for (a,b) in flat_tagged_words]

    # get the conditional frequency distribution for the tagged words
    tagged_cfd = nltk.ConditionalFreqDist(rev_tagged_words)

    #get the possible parts-of-speech
    tag = nltk.data.load('help/tagsets/upenn_tagset.pickle')

    # print the conditional distribution in a file
    sys.stdout = open(corpus_name + '-pos-word-freq.txt', 'w')
    print(tagged_cfd.tabulate())
    sys.stdout = sys.__stdout__

    """
    Similar words
    """
    #get the raw text in a variable for finding similar words
    text = nltk.Text(l_flat_words)

    # print most frequent words for the different POS(NN, VBD, JJ, RB)
    most_common_nn = tagged_cfd['NN'].most_common()[0][0]
    print('The most frequent word in the POS NN is', most_common_nn, 'and its similar words are:')
    print(text.similar(most_common_nn))
    most_common_vbd = tagged_cfd['VBD'].most_common()[0][0]
    print('The most frequent word in the POS VBD is', most_common_vbd, 'and its similar words are:')
    print(text.similar(most_common_vbd))
    most_common_jj = tagged_cfd['JJ'].most_common()[0][0]
    print('The most frequent word in the POS JJ is', most_common_jj, 'and its similar words are:')
    print(text.similar(most_common_jj))
    most_common_rb = tagged_cfd['RB'].most_common()[0][0]
    print('The most frequent word in the POS RB is', most_common_rb, 'and its similar words are:')
    print(text.similar(most_common_rb))


    """
    Collocations
    """

    #print the collocations of the text
    print('Collocations:')
    print(text.collocations())
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
        