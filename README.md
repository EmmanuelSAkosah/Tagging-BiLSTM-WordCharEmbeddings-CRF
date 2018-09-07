#NLP with Deep Learning: Tagging with bi-LSTM based models
This project is an adaptation of Guillaume's [model](https://github.com/guillaumegenthial/sequence_tagging) for NER.
I adapted the model to have more configurable options and to solve
more tasks. The additional configurations were added to investigate how they
impact performance of the model. 
Documentation, which includes theoretical description of the model and comparison of the different model configurations , can be found [here](https://docs.google.com/document/d/1jxlbAKcsZcQvHh4u2prLgnfnNJMH4AJwVxf-i5xwtoM/edit?usp=sharing).

###Contribution:
1. Added multitasking: This version solves POS ,CHUNKING and NER model
2. Added word2vec: choose between google's word2vec pre-trained word embedding or Stanford's of gloVe word embedding. 
One can compare the resulting accuracy
3. Added CNN-char embedding: (a terrible bug persists that needs fixing) 

###How to Run:
1. In the terminal run
$ `make wembedding` 
to download the standford glove file and 
$ `./word2vec_download_google_model.sh`
to download google news word2vec file

2. Run build_data.py to generate the required files for training.
This generates .txt word embedding files and trimmed versions of them
Note:The process of writing word2vec to a .txt file takes might take
long time depending on your machine. when written , the file in over 10GB in size
the process takes about 15 mins on a six-core 5.0GHz  machine. 

3. Configure model in config.py.
Here, you can configure the model to perform one task from POS, CHUNK and NER.
You can specify to whether to use word_embedding or not and which type to use, whether to use CRF, character 
embedding etc, and tune a number of hyperparameters. 
Note: running build.data.py generates a tag-specific file. 
So for example, to perform NER one must re-run build.data.py 
with 'task' in config.py set to NER in order to generate the required tag file.

4. Run train.py to train the model.







