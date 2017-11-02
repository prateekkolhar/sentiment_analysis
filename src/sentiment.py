import sys
import models as md
import sentiment_data as sd
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    # Use either 50-dim or 300-dim vectors
    #word_vectors = read_word_embeddings("data/glove.6B.50d-relativized.txt")
    word_vectors = sd.read_word_embeddings("data/glove.6B.300d-relativized.txt")

    # Load train, dev, and test exs
    train_exs = sd.read_and_index_sentiment_examples("data/train.txt", word_vectors.word_indexer)
    dev_exs = sd.read_and_index_sentiment_examples("data/dev.txt", word_vectors.word_indexer)
    test_exs = sd.read_and_index_sentiment_examples("data/test-blind.txt", word_vectors.word_indexer)
    print repr(len(train_exs)) + " / " + repr(len(dev_exs)) + " / " + repr(len(test_exs)) + " train/dev/test examples"

    # system_to_run = "FANCY"
    if len(sys.argv) >= 2:
        system_to_run = sys.argv[1]
    else:
        system_to_run = "FANCY"
        
    if system_to_run == "FF":
        with tf.variable_scope("model", reuse=None):
            test_exs_predicted = md.train_ffnn(train_exs, dev_exs, test_exs, word_vectors,"ffnn", 10)
#         write_sentiment_examples(test_exs_predicted, "test-blind.output.txt", word_vectors.word_indexer)
    elif system_to_run == "FANCY":
        with tf.variable_scope("model2", reuse=None):
            test_exs_predicted = md.train_bi_lstm(train_exs, dev_exs, test_exs, word_vectors, "bi-lstm", 10)
    else:
        raise Exception("Pass in either FF or FANCY to run the appropriate system")
    # Write the test set output
    sd.write_sentiment_examples(test_exs_predicted, "test-blind.output.txt", word_vectors.word_indexer)