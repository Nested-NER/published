"""Transfer glove word embedding to gensim format"""

import os
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# input file
glove_file = datapath(os.getcwd() + '/model/word2vec/glove.6B.100d.txt')
# output file
save_file = get_tmpfile(os.getcwd() + "/model/word2vec/glove_word2vec.txt")

glove2word2vec(glove_file, save_file)

# model = KeyedVectors.load_word2vec_format(tmp_file)
