from model.config import Config
from model.data_utils import CoNLLDataset, get_vocabs, UNK, NUM, \
    get_glove_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_glove_vectors, get_processing_word,\
    export_trimmed_word2vec_vectors,write_word2vec_to_txtfile


def main():
    """Procedure to build data

    You MUST RUN this procedure. It iterates over the whole dataset (train,
    dev and test) and extract the vocabularies in terms of words, tags, and
    characters. Having built the vocabularies it writes them in a file. The
    writing of vocabulary in a file assigns an id (the line #) to each word.
    It then extract the relevant GloVe vectors and stores them in a np array
    such that the i-th entry corresponds to the i-th word in the vocabulary.


    Args:
        config: (instance of Config) has attributes like hyper-params...

    """
    # get config and processing of words
    config = Config(load=False)
    processing_word = get_processing_word(lowercase=True)

    # Generators
    dev   = CoNLLDataset(config.filename_dev, processing_word, task= config.task)
    test  = CoNLLDataset(config.filename_test, processing_word, task= config.task)
    train = CoNLLDataset(config.filename_train, processing_word, task=config.task)


    # Build Word and Tag vocab
    vocab_words, vocab_tags = get_vocabs([train, dev, test])
    vocab_glove = get_glove_vocab(config.filename_glove)
    #TODO get word2vec vocab too

    vocab = vocab_words & vocab_glove
    vocab.add(UNK)
    vocab.add(NUM)

    # Save word and tag vocab
    write_vocab(vocab, config.filename_words)
    write_vocab(vocab_tags, config.filename_tags)

    # write and trim GloVe and word2vec Vectors
    vocab = load_vocab(config.filename_words)
    write_word2vec_to_txtfile(config.path_to_word2vec_bin_file, config.filename_word2vec)
    export_trimmed_word2vec_vectors(vocab, config.filename_word2vec,
                                    config.trimmed_word2vec_filename, config.dim_word)

    export_trimmed_glove_vectors(vocab, config.filename_glove,
                                 config.trimmed_glove_filename, config.dim_word)


    # Build and save char vocab
    train = CoNLLDataset(config.filename_train)
    vocab_chars = get_char_vocab(train)
    write_vocab(vocab_chars, config.filename_chars)



if __name__ == "__main__":
    main()