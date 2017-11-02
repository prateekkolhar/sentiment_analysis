# Analogous to train_ffnn, but trains your fancier model.
def train_bi_lstm_batch(train_exs, dev_exs, test_exs, word_vectors, num_epochs=-1):
    
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
    
    data = tf.placeholder(tf.float64, [batch_size, seq_max_len, feat_vec_size] ,name="data")
    seq_len = tf.placeholder(tf.float64, batch_size ,name="seq_len")
    
   
    lstmCell_fw = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmCell_fw = tf.contrib.rnn.DropoutWrapper(cell=lstmCell_fw, output_keep_prob=0.75)
    
    lstmCell_bw = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmCell_bw = tf.contrib.rnn.DropoutWrapper(cell=lstmCell_bw, output_keep_prob=0.75)
    
    # lstmCell_fw = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    # lstmCell_fw = tf.contrib.rnn.DropoutWrapper(cell=lstmCell_fw, output_keep_prob=0.75)
    # value, _ = tf.nn.dynamic_rnn(lstmCell_fw, data, dtype=tf.float64)

    (value, _) = tf.nn.bidirectional_dynamic_rnn(lstmCell_fw, lstmCell_bw, data, dtype=tf.float64, time_major = False)
    
    
    
    
#     fx = value[0][-1]
    
#     V = tf.get_variable("V", [embedding_size, feat_vec_size], initializer=tf.contrib.layers.xavier_initializer(seed=0), dtype=tf.float64)
    
    z = tf.concat([tf.transpose(value[0],[1,0,2])[-1],tf.transpose(value[1],[1,0,2])[-1]],1)
    W = tf.get_variable("W", [num_classes, lstmUnits*2], initializer=tf.contrib.layers.xavier_initializer(seed=0), dtype=tf.float64)
    probs = tf.nn.softmax(tf.tensordot(W, tf.transpose(z), 1))
    
    one_best = tf.argmax(probs,0)

   
    label = tf.placeholder(tf.int32, batch_size, name="label")
    
    label_onehot = tf.one_hot(label, num_classes, dtype=tf.float64)
    loss = tf.reduce_sum(tf.negative( tf.multiply( tf.transpose( label_onehot ) , tf.log(probs) ) ) )


    
    decay_steps = 100
    learning_rate_decay_factor = 0.999995
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
    # opt = tf.train.AdamOptimizer(lr)
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
        print "dev set size : "+repr(len(dev_ys))
        for i in range(0, num_epochs):
            
            loss_this_iter = 0
            # batch_size of 1 here; if we want bigger batches, we need to build our network appropriately
            print "epochs:"+str(i+1)
            for ex_idx in xrange(0, len(train_xs)-batch_size, batch_size):
                # sess.run generally evaluates variables in the computation graph given inputs. "Evaluating" train_op
                # causes training to happen
                [_, loss_this_instance, summary] = sess.run([train_op, loss, merged], feed_dict = {data: train_xs[ex_idx:ex_idx+batch_size],
                                                                      label: train_ys[ex_idx:ex_idx+batch_size],
                                                                      seq_len: train_seq_lens[ex_idx:ex_idx+batch_size]})
                train_writer.add_summary(summary, step_idx)
                step_idx += 1
                loss_this_iter += loss_this_instance
            print "Train loss : " + repr(loss_this_iter)
            
            def evaluate_examples(xs, ys, seq_lens):
                correct = 0
                for ex_idx in xrange(0, len(xs)):
                    [probs_this_instance, pred_this_instance, z_this_instance] = sess.run([probs, one_best, z],
                                                                          feed_dict={data: [xs[ex_idx] for i in xrange(batch_size)],
                                                                      seq_len: np.array([seq_lens[ex_idx]for i in xrange(batch_size)]) })
                    probs_this_instance, pred_this_instance, z_this_instance = probs_this_instance[0], pred_this_instance[0], z_this_instance[0]
                    if (ys[ex_idx] == pred_this_instance):
                        correct += 1
                print "Dev accuracy : "+repr(correct*1.0/len(ys))
            evaluate_examples(dev_xs,dev_ys,dev_seq_lens)
        stop = timeit.default_timer()
        print "Total Time:" + str(stop-start)
