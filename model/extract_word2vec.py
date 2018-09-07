
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

# this line doesn't load the trained model
from gensim.models.keyedvectors import KeyedVectors


def main():
    words = ['access', 'aeroway', 'airport']
    import numpy as np
    #data = np.load("../data/word2vec/word2vec_300d_trimmed.txt.npz")
    data = np.load("../data/glove.6B.300d.trimmed.npz")

    vectors = data.files[0]
    print(len(vectors))

    """
    

        filename = "../data/word2vec/word2vec_300d.txt"
        path_to_model = "../data/word2vec/GoogleNews-vectors-negative300.bin"
        # this is how you load the model
        print("Loading model...")
        model = KeyedVectors.load_word2vec_format(path_to_model, binary=True)
        print("Saving model...")
        model.wv.save_word2vec_format(filename)
        print("Done saving model. YAY")
        # to extract word vector
        print(model[words[0]])  #access
        
    """

if __name__ == "__main__":
    main()