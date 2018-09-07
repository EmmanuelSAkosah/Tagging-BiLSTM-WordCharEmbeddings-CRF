import os

from enum import Enum
from .general_utils import get_logger
from .data_utils import get_trimmed_word_embedding_vectors, load_vocab, \
        get_processing_word

class Config():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()



    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags  = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)

        self.nwords     = len(self.vocab_words)
        self.nchars     = len(self.vocab_chars)
        self.ntags      = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,
                self.vocab_chars, lowercase=True, chars=self.use_chars)
        self.processing_tag  = get_processing_word(self.vocab_tags,
                lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_word_embedding_vectors(self.trimmed_word_embedding_filename)
                if self.use_pretrained else None)


    tasks = {
        'POS'   : 1,
        'CHUNK' : 2,
        'NER'   : 3
    }


                                 ####Configurations####
    # set task type POS, CHUNK, NER
    task = tasks['POS']

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True # if crf, training is 1.7x slower on CPU
    use_chars = True # if false, model doesn't include charcater embeddings. char embedding, training is 3.5x slower on CPU
    use_cnn_char_embedding = False #, if true, cnn char embeddings are used instead of bilstm
    use_word2vec = True  # if true, replace GloVe embedding with gensim word2vec


    # model output files
    dir_output = "results"
    dir_model  = dir_output + "model.weights/"
    path_log   = dir_output + "log.txt"

    # embeddings
    dim_word = 300
    dim_char = 100

    # word embedding files
    use_pretrained = True
    filename_glove = "data/glove.6B/glove.6B.{}d.txt".format(dim_word)
    #trimmed embeddings (created from glove_filename with build_data.py)
    trimmed_glove_filename = "data/glove.6B/glove.6B.{}d.trimmed.npz".format(dim_word)

    #word2vec files
    path_to_word2vec_bin_file = "data/word2vec/GoogleNews-vectors-negative300.bin"
    trimmed_word2vec_filename = "data/word2vec/word2vec_300d_trimmed.npz".format(dim_word)
    filename_word2vec = "data/word2vec/word2vec_300d.txt"

    if use_word2vec == True:
        word_embedding_filename = filename_word2vec
        trimmed_word_embedding_filename = trimmed_word2vec_filename
    else:
        word_embedding_filename = filename_glove
        trimmed_word_embedding_filename = trimmed_glove_filename


        # dataset
    filename_dev = "data/CoNLL2003_eng_test.txt"
    filename_test = "data/test.txt"
    filename_train = "data/CoNLL2003_eng_train.txt"

    max_iter = None # if not None, max number of examples in Dataset

    # word vocab, character set and tag set (created from dataset with build_data.py)
    filename_words = "data/words.txt"
    if task == tasks['POS']:
        filename_tags = "data/POS_tags.txt"
    elif task == tasks['NER']:
        filename_tags = "data/NER_tags.txt"
    else:
       filename_tags = "data/CHUNK_tags.txt"
    filename_chars = "data/chars.txt"

    # training
    train_embeddings = False
    nepochs          = 15
    dropout          = 0.5
    batch_size       = 20
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = -1 # if negative, no clipping
    nepoch_no_imprv  = 3

    #char cnn
    max_word_length = 50
    filter_size = [2, 3, 4, 5]
    num_of_filters = 128


    # model hyperparameters
    hidden_size_char = 100 # lstm on chars
    hidden_size_lstm = 300 # lstm on word embeddings
