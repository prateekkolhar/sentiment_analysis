# models.py

import tensorflow as tf
import numpy as np
import random
from sentiment_data import *
import timeit


# Returns a new numpy array with the data from np_arr padded to be of length length. If length is less than the
# length of the base array, truncates instead.
def pad_to_length(np_arr, length):
    result = np.zeros(length)
    result[0:np_arr.shape[0]] = np_arr
    return result


# Train a feedforward neural network on the given training examples, using dev_exs for development and returning
# predictions on the *blind* test_exs (all test_exs have label 0 as a dummy placeholder value). Returned predictions
# should be SentimentExample objects with predicted labels and the same sentences as input (but these won't be
# read for evaluation anyway)
def train_ffnn(train_exs, dev_exs, test_exs, word_vectors, num_epochs=-1):
    # 59 is the max sentence length in the corpus, so let's set this to 60
    
    seq_max_len = 60
    
    def generate_feature_mat(examples):
#         mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in examples])
#         seq_lens = np.array([len(ex.indexed_words) for ex in examples])
#         labels_arr = np.array([ex.label for ex in examples])
#         full_mat = np.ones([mat.shape[0],mat.shape[1],word_vectors.vectors.shape[1]])
        
#         for i in xrange(mat.shape[0]):
#             for j in xrange(mat.shape[1]):
#                 full_mat[i][j]=word_vectors.get_embedding_from_idx(int(mat[i][j]))
          
#         xs = full_mat.mean(1)        
        ys = np.array([ex.label for ex in examples])
#         Can also send the seq_lens, labels_arr
        
        xs = np.zeros([len(examples),word_vectors.vectors.shape[1]])
        for i in xrange(len(examples)):
            word_idxs = np.array(examples[i].indexed_words)
            all_words = np.ones([len(word_idxs), word_vectors.vectors.shape[1]])*-1
            for j in xrange(len(word_idxs)):
                all_words[j] = word_vectors.get_embedding_from_idx(int(word_idxs[j]))
            xs[i]=all_words.mean(0) 
    
        return(xs,ys)

    (train_xs,train_ys) = generate_feature_mat(train_exs)
    (dev_xs,dev_ys) = generate_feature_mat(dev_exs)
    
    print train_xs.shape
    print type(train_xs[0][0])
    print dev_xs.shape

    feat_vec_size = word_vectors.vectors.shape[1]
    embedding_size = feat_vec_size
    num_classes = 2
    ffnn_keep_prob1 = tf.placeholder(tf.float64, 1, name="ffnn_keep_probability1")
    ffnn_keep_prob2 = tf.placeholder(tf.float64, 1, name="ffnn_keep_probability2")
    fx = tf.placeholder(tf.float64, feat_vec_size)
    
    V = tf.get_variable("V", [embedding_size, feat_vec_size], initializer=tf.contrib.layers.xavier_initializer(seed=0), dtype=tf.float64)
     
    fx1 = tf.nn.relu(tf.tensordot(V, tf.nn.dropout(fx,ffnn_keep_prob1[0]), 1))
    
    V1 = tf.get_variable("V1", [embedding_size, embedding_size], initializer=tf.contrib.layers.xavier_initializer(seed=0), dtype=tf.float64)
    
    z = tf.nn.relu(tf.tensordot(V1, tf.nn.dropout(fx1,ffnn_keep_prob2[0]), 1))
    
    # z = tf.sigmoid(tf.tensordot(V, fx, 1))
    
    W = tf.get_variable("W", [num_classes, embedding_size], dtype=tf.float64)
    probs = tf.nn.softmax(tf.tensordot(W, z, 1))
    
    one_best = tf.argmax(probs)

   
    label = tf.placeholder(tf.int32, 1)
    
    label_onehot = tf.reshape(tf.one_hot(label, num_classes, dtype=tf.float64), shape=[num_classes])
    loss = tf.negative(tf.log(tf.tensordot(probs, label_onehot, 1)))


    
    decay_steps = 10
    learning_rate_decay_factor = .999995
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
    initial_learning_rate = 0.0005
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    # Logging with Tensorboard
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('loss', loss)
    # Plug in any first-order method here! We'll use Adam, which works pretty well, but SGD with momentum, Adadelta,
    # and lots of other methods work well too
    opt = tf.train.AdamOptimizer(lr)
    # Loss is the thing that we're optimizing
    grads = opt.compute_gradients(loss)
    # Now that we have gradients, we operationalize them by defining an operator that actually applies them.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    # RUN TRAINING AND TEST
    # Initializer; we need to run this first to initialize variables
    init = tf.global_variables_initializer()
    if num_epochs==-1: num_epochs = 10
    merged = tf.summary.merge_all()  # merge all the tensorboard variables
    # The computation graph must be run in a particular Tensorflow "session". Parameters, etc. are localized to the
    # session (unless you pass them around outside it). All runs of a computation graph with certain values are relative
    # to a particular session
    with tf.Session() as sess:
        # Write a logfile to the logs/ directory, can use Tensorboard to view this
        train_writer = tf.summary.FileWriter('logs/', sess.graph)
        # Generally want to determinize training as much as possible
        tf.set_random_seed(0)
        # Initialize variables
        sess.run(init)
        step_idx = 0
        for i in range(0, num_epochs):
            loss_this_iter = 0
            # batch_size of 1 here; if we want bigger batches, we need to build our network appropriately
            for ex_idx in xrange(0, len(train_xs)):
                # sess.run generally evaluates variables in the computation graph given inputs. "Evaluating" train_op
                # causes training to happen
                [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {fx: train_xs[ex_idx],
                                                                                  label: np.array([train_ys[ex_idx]]),
                                                                                                  ffnn_keep_prob1 : [.8],
                                                                                                  ffnn_keep_prob2 : [.5]})
                train_writer.add_summary(summary, step_idx)
                step_idx += 1
                loss_this_iter += loss_this_instance
            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)
            
            def evaluate_examples(xs, ys):
                correct = 0
                for ex_idx in xrange(0, len(xs)):
                    [probs_this_instance, pred_this_instance, z_this_instance] = sess.run([probs, one_best, z],
                                                                                          feed_dict={fx: xs[ex_idx],
                                                                                                    ffnn_keep_prob1 : [1.0],
                                                                                                    ffnn_keep_prob2 : [1.0]})
                    if (ys[ex_idx] == pred_this_instance):
                        correct += 1
                print repr(correct) + "/" + repr(len(ys)) + " correct after training"
                print repr(correct*1.0/len(ys)) + " correct after training"
            evaluate_examples(dev_xs,dev_ys)


def train_bi_lstm(train_exs, dev_exs, test_exs, word_vectors, file_desc, num_epochs=-1):
    
    # 59 is the max sentence length in the corpus, so let's set this to 60
    
    seq_max_len = 60
    
    def generate_feature_mat(examples):
        mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in examples])
        seq_lens = np.array([len(ex.indexed_words) for ex in examples])
        labels_arr = np.array([ex.label for ex in examples])
        full_mat = np.ones([mat.shape[0],mat.shape[1],word_vectors.vectors.shape[1]])
        
        for i in xrange(mat.shape[0]):
            for j in xrange(mat.shape[1]):
                full_mat[i][j]=word_vectors.get_embedding_from_idx(int(mat[i][j]))
        xs = full_mat  
        ys = np.array([ex.label for ex in examples])
#         Can also send the seq_lens, labels_arr
        return(xs, ys, seq_lens)

    (train_xs,train_ys, train_seq_lens) = generate_feature_mat(train_exs)
    (dev_xs,dev_ys,dev_seq_lens) = generate_feature_mat(dev_exs)
    (test_xs,test_ys,test_seq_lens) = generate_feature_mat(test_exs)
    
    print train_xs.shape
    print type(train_xs[0][0])
    print dev_xs.shape

    feat_vec_size = word_vectors.vectors.shape[1]
    embedding_size = feat_vec_size
    num_classes = 2
    lstmUnits = 100
    batch_size = 1;
    
    data = tf.placeholder(tf.float64, [1, seq_max_len, feat_vec_size] ,name="data")
    seq_len = tf.placeholder(tf.int32, 1 ,name="seq_len")
    
    keep_prob = tf.placeholder(tf.float64, 1, name="keep_probability")

    lstmCell_fw = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmCell_fw = tf.contrib.rnn.DropoutWrapper(cell=lstmCell_fw, output_keep_prob=keep_prob[0])
    lstmCell_bw = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmCell_bw = tf.contrib.rnn.DropoutWrapper(cell=lstmCell_bw, output_keep_prob=keep_prob[0])
    
    # lstmCell_fw = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    # lstmCell_fw = tf.contrib.rnn.DropoutWrapper(cell=lstmCell_fw, output_keep_prob=0.75)
    # value, _ = tf.nn.dynamic_rnn(lstmCell_fw, data, dtype=tf.float64)

    (value, _) = tf.nn.bidirectional_dynamic_rnn(lstmCell_fw, lstmCell_bw, data, dtype=tf.float64, time_major = False)
    ffnn_keep_prob = tf.placeholder(tf.float64, 1, name="ffnn_keep_probability")
    
#     fx = tf.reduce_mean(tf.concat( [ value[0][0][:seq_len[0]] , value[1][0][:seq_len[0]] ],1),0)
    
#     V = tf.get_variable("V", [lstmUnits*2, lstmUnits*2], initializer=tf.contrib.layers.xavier_initializer(seed=0), dtype=tf.float64)
    
#     fx1 = tf.sigmoid(tf.tensordot(V, tf.nn.dropout(fx,ffnn_keep_prob[0]), 1))
    
#     V1 = tf.get_variable("V1", [lstmUnits*2, lstmUnits*2], initializer=tf.contrib.layers.xavier_initializer(seed=0), dtype=tf.float64)
    
    
    
    
#     z = tf.sigmoid(tf.tensordot(V1, tf.nn.dropout(fx1,ffnn_keep_prob[0]), 1))
    z = tf.reduce_mean(tf.concat( [ value[0][0][:seq_len[0]] , value[1][0][:seq_len[0]] ],1),0) 
    # z = tf.concat(
    #     [
    #         tf.reduce_mean(tf.concat( [ value[0][0][:seq_len[0]] , value[1][0][:seq_len[0]] ],1),0),
    #         tf.reduce_max(tf.concat( [ value[0][0][:seq_len[0]] , value[1][0][:seq_len[0]] ],1),0) 
    #     ],
    #     0
    # )
    W = tf.get_variable("W", [num_classes, lstmUnits*2],dtype=tf.float64)
    probs = tf.nn.softmax(tf.tensordot(W, z, 1))
    
    one_best = tf.argmax(probs)

   
    label = tf.placeholder(tf.int32, 1)
    
    label_onehot = tf.reshape(tf.one_hot(label, num_classes, dtype=tf.float64), shape=[num_classes])
    loss = tf.negative(tf.log(tf.tensordot(probs, label_onehot, 1)))


    
    decay_steps = 10
    learning_rate_decay_factor = 0.9995
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
    initial_learning_rate = 0.01
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    # Logging with Tensorboard
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('loss', loss)
    # Plug in any first-order method here! We'll use Adam, which works pretty well, but SGD with momentum, Adadelta,
    # and lots of other methods work well too
    opt = tf.train.AdamOptimizer()
    # Loss is the thing that we're optimizing
    grads = opt.compute_gradients(loss)
    # Now that we have gradients, we operationalize them by defining an operator that actually applies them.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    # RUN TRAINING AND TEST
    # Initializer; we need to run this first to initialize variables
    init = tf.global_variables_initializer()
    if num_epochs==-1: num_epochs = 10
    merged = tf.summary.merge_all()  # merge all the tensorboard variables
    # The computation graph must be run in a particular Tensorflow "session". Parameters, etc. are localized to the
    # session (unless you pass them around outside it). All runs of a computation graph with certain values are relative
    # to a particular session
    with tf.Session() as sess:
        # Write a logfile to the logs/ directory, can use Tensorboard to view this
        train_writer = tf.summary.FileWriter('logs/', sess.graph)
        # Generally want to determinize training as much as possible
        tf.set_random_seed(0)
        # Initialize variables
        sess.run(init)
        step_idx = 0
        start = timeit.default_timer()
        for i in range(0, num_epochs):
            loss_this_iter = 0
            # batch_size of 1 here; if we want bigger batches, we need to build our network appropriately
            for ex_idx in xrange(0, len(train_xs)):
                # sess.run generally evaluates variables in the computation graph given inputs. "Evaluating" train_op
                # causes training to happen
                [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {data: [train_xs[ex_idx]],
                                                                                  label: np.array([train_ys[ex_idx]]),
                                                                                  seq_len: [int(train_seq_lens[ex_idx])],
                                                                                                  keep_prob: [.75],
                                                                                                  ffnn_keep_prob : [.75]})
                train_writer.add_summary(summary, step_idx)
                step_idx += 1
                loss_this_iter += loss_this_instance
            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)
            
            def evaluate_examples(xs, ys, seq_lens, print_acc, test_exs=-1, ):
                correct = 0
                pred=np.zeros(len(ys))
                for ex_idx in xrange(0, len(xs)):
                    [probs_this_instance, pred_this_instance, z_this_instance] = sess.run([probs, one_best, z],
                                                                                          feed_dict={data: [xs[ex_idx]],
                                                                                                    seq_len: [int(seq_lens[ex_idx])],
                                                                                                    keep_prob: [1.0],
                                                                                                    ffnn_keep_prob : [1.0]})
                    if (ys[ex_idx] == pred_this_instance):
                        correct += 1
                    pred[ex_idx]=pred_this_instance
                    
                    if test_exs!=-1:
                        test_exs[ex_idx].label=pred_this_instance
                accuracy = correct*1.0/len(ys)
                if print_acc:
                    print repr(correct) + "/" + repr(len(ys)) + " correct after training"
                    print repr(accuracy) + " correct after training"
                
                
                return test_exs, accuracy
            _,accuracy = evaluate_examples(dev_xs,dev_ys, dev_seq_lens, True)
            test_exs_predicted,_ = evaluate_examples(test_xs,test_ys, test_seq_lens, False, test_exs)
            
            write_sentiment_examples(test_exs_predicted, "./output/"+file_desc+"_e"+str(i)+"_a"+str(round(accuracy,3))+".output.txt", word_vectors.word_indexer)
        stop = timeit.default_timer()
        print "Total Time:" + str(stop-start)


        
def train_lstm(train_exs, dev_exs, test_exs, word_vectors, num_epochs=-1):
    
    # 59 is the max sentence length in the corpus, so let's set this to 60
    
    seq_max_len = 60
    
    def generate_feature_mat(examples):
        mat = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in examples])
        seq_lens = np.array([len(ex.indexed_words) for ex in examples])
        labels_arr = np.array([ex.label for ex in examples])
        full_mat = np.ones([mat.shape[0],mat.shape[1],word_vectors.vectors.shape[1]])
        
        for i in xrange(mat.shape[0]):
            for j in xrange(mat.shape[1]):
                full_mat[i][j]=word_vectors.get_embedding_from_idx(int(mat[i][j]))
        xs = full_mat  
        ys = np.array([ex.label for ex in examples])
#         Can also send the seq_lens, labels_arr
        return(xs, ys, seq_lens)

    (train_xs,train_ys, train_seq_lens) = generate_feature_mat(train_exs)
    (dev_xs,dev_ys,dev_seq_lens) = generate_feature_mat(dev_exs)
    
    print train_xs.shape
    print type(train_xs[0][0])
    print dev_xs.shape

    feat_vec_size = word_vectors.vectors.shape[1]
    embedding_size = feat_vec_size
    num_classes = 2
    lstmUnits = 100
    batch_size = 1;
    
    data = tf.placeholder(tf.float64, [1, seq_max_len, feat_vec_size] ,name="data")
    seq_len = tf.placeholder(tf.int32, 1 ,name="seq_len")
    
    keep_prob = tf.placeholder(tf.float64, 1 ,name="keep_probability")
    lstmCell_fw = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmCell_fw = tf.contrib.rnn.DropoutWrapper(cell=lstmCell_fw, output_keep_prob=keep_prob[0])
    
    value, _ = tf.nn.dynamic_rnn(lstmCell_fw, data, dtype=tf.float64)

    
    
    
    
    
#     fx = value[0][-1]
    
#     V = tf.get_variable("V", [embedding_size, feat_vec_size], initializer=tf.contrib.layers.xavier_initializer(seed=0), dtype=tf.float64)
    
    z = value[0][seq_len[0]-1]
    W = tf.get_variable("W", [num_classes, lstmUnits], dtype=tf.float64)
    probs = tf.nn.softmax(tf.tensordot(W, z, 1))
    
    one_best = tf.argmax(probs)

   
    label = tf.placeholder(tf.int32, 1)
    
    label_onehot = tf.reshape(tf.one_hot(label, num_classes, dtype=tf.float64), shape=[num_classes])
    loss = tf.negative(tf.log(tf.tensordot(probs, label_onehot, 1)))


    
    decay_steps = 10
    learning_rate_decay_factor = 0.995
    global_step = tf.contrib.framework.get_or_create_global_step()
    # Smaller learning rates are sometimes necessary for larger networks
    initial_learning_rate = 0.005
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    # Logging with Tensorboard
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('loss', loss)
    # Plug in any first-order method here! We'll use Adam, which works pretty well, but SGD with momentum, Adadelta,
    # and lots of other methods work well too
    opt = tf.train.AdamOptimizer(learning_rate = lr)
    # Loss is the thing that we're optimizing
    grads = opt.compute_gradients(loss)
    # Now that we have gradients, we operationalize them by defining an operator that actually applies them.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    # RUN TRAINING AND TEST
    # Initializer; we need to run this first to initialize variables
    init = tf.global_variables_initializer()
    if num_epochs==-1: num_epochs = 10
    merged = tf.summary.merge_all()  # merge all the tensorboard variables
    # The computation graph must be run in a particular Tensorflow "session". Parameters, etc. are localized to the
    # session (unless you pass them around outside it). All runs of a computation graph with certain values are relative
    # to a particular session
    with tf.Session() as sess:
        # Write a logfile to the logs/ directory, can use Tensorboard to view this
        train_writer = tf.summary.FileWriter('logs/', sess.graph)
        # Generally want to determinize training as much as possible
        tf.set_random_seed(0)
        # Initialize variables
        sess.run(init)
        step_idx = 0
        start = timeit.default_timer()
        for i in range(0, num_epochs):
            loss_this_iter = 0
            # batch_size of 1 here; if we want bigger batches, we need to build our network appropriately
            for ex_idx in xrange(0, len(train_xs)):
                # sess.run generally evaluates variables in the computation graph given inputs. "Evaluating" train_op
                # causes training to happen
                [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {data: [train_xs[ex_idx]],
                                                                                  label: np.array([train_ys[ex_idx]]),
                                                                                  seq_len: [int(train_seq_lens[ex_idx])],
                                                                                    keep_prob: [0.75]})
                train_writer.add_summary(summary, step_idx)
                step_idx += 1
                loss_this_iter += loss_this_instance
            print "Loss for iteration " + repr(i) + ": " + repr(loss_this_iter)
            
            def evaluate_examples(xs, ys, seq_lens):
                correct = 0
                for ex_idx in xrange(0, len(xs)):
                    [probs_this_instance, pred_this_instance, z_this_instance] = sess.run([probs, one_best, z],
                                                                                          feed_dict={data: [xs[ex_idx]],
                                                                                                    seq_len: [int(seq_lens[ex_idx])],
                                                                                                    keep_prob: [1.0]})
                    if (ys[ex_idx] == pred_this_instance):
                        correct += 1
                print repr(correct) + "/" + repr(len(ys)) + " correct after training"
                print repr(correct*1.0/len(ys)) + " correct after training"
            evaluate_examples(dev_xs,dev_ys, dev_seq_lens)
        stop = timeit.default_timer()
        print "Total Time:" + str(stop-start)
