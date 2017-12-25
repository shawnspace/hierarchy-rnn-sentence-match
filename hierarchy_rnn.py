import tensorflow as tf
import helper

cell_types = {
    'BasicRNN':tf.nn.rnn_cell.BasicRNNCell,
    'BasicLSTM':tf.nn.rnn_cell.BasicLSTMCell
}

def get_embeddings(hparams):
    if hparams.glove_path and hparams.vocab_path:
        tf.logging.info("Loading Glove embeddings...")
        vocab_array, vocab_dict = helper.load_vocab(hparams.vocab_path)
        glove_vectors, glove_dict = helper.load_glove_vectors(hparams.glove_path, vocab=set(vocab_array))
        initializer = helper.build_initial_embedding_matrix(vocab_dict, glove_dict, glove_vectors,
                                                             hparams.embedding_dim)
    else:
        tf.logging.info("No glove/vocab path specificed, starting with random embeddings.")
        initializer = tf.random_uniform_initializer(-0.25, 0.25)

    return tf.get_variable(
        "word_embeddings",
        initializer=initializer,
        trainable=False)

def hierarchy_rnn(hparams, mode, contexts, contexts_len, query, query_len, label):

    embedding_W = get_embeddings(hparams)

    query_embeded = tf.nn.embedding_lookup(embedding_W,query)
    contexts_embeded = []
    for v in contexts:
        contexts_embeded.append(tf.nn.embedding_lookup(embedding_W, v))

    with tf.variable_scope('sentence_rnn') as vs:
        sentence_cell = (cell_types[hparams.rnn_cell_type])(num_units=hparams.sentence_level_encoding_num_units)
        if hparams.keep_prob <1.0:
            sentence_cell = tf.contrib.rnn.DropoutWrapper(sentence_cell,state_keep_prob=hparams.keep_prob)
        contexts_encoding = []#[batch_size, sentence_level_encoding_num_units]
        for i, ce in enumerate(contexts_embeded):
            output,state = tf.nn.dynamic_rnn(sentence_cell, ce, sequence_length=contexts_len[i],dtype=tf.float32)
            if hparams.rnn_cell_type == 'BasicRNN':
                contexts_encoding.append(state)
            elif hparams.rnn_cell_type == 'BasicLSTM':
                contexts_encoding.append(state.h)
        output, state = tf.nn.dynamic_rnn(sentence_cell, query_embeded, query_len, dtype=tf.float32)
        if hparams.rnn_cell_type == 'BasicRNN':
            query_encoding = state
        elif hparams.rnn_cell_type == 'BasicLSTM':
            query_encoding = state.h

    with tf.variable_scope('context_rnn') as vs:
        context_cell = (cell_types[hparams.rnn_cell_type])(num_units=hparams.session_level_encoding_num_units)
        if hparams.keep_prob <1.0:
            context_cell = tf.contrib.rnn.DropoutWrapper(context_cell,state_keep_prob=hparams.keep_prob)
        contexts_encoding_expand = [tf.expand_dims(c, axis=1) for c in contexts_encoding]
        contexts_encoding_merge = tf.concat(contexts_encoding_expand, axis=1)# shape = [batch_size, context_width, wv_dim]
        output, state = tf.nn.dynamic_rnn(context_cell, contexts_encoding_merge, dtype=tf.float32)
        if hparams.rnn_cell_type == 'BasicRNN':
            context_encoding = state
        elif hparams.rnn_cell_type =='BasicLSTM':
            context_encoding = state.h

    with tf.variable_scope('prediction') as vs:
        W = tf.get_variable('prediction_w', shape=[hparams.sentence_level_encoding_num_units, hparams.session_level_encoding_num_units],initializer=tf.truncated_normal_initializer())
        qW = tf.matmul(query_encoding,W) #shape=[batch_size, session_level_encoding_num_units]
        qW = tf.expand_dims(qW, axis=2)  #shape=[batch_size, session_level_encoding_num_units, 1]
        context_encoding = tf.expand_dims(context_encoding, axis=2) #shape=[batch_size, session_level_encoding_num_units, 1]
        qWc = tf.matmul(qW, context_encoding, transpose_a=True) #shape= [batch_size, 1,1]
        logits = tf.squeeze(qWc,axis=[1,2])
        probs = tf.nn.sigmoid(logits)

        if mode == tf.contrib.learn.ModeKeys.INFER:
            return probs, 0.0

        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(label), logits=logits)

    mean_loss = tf.reduce_mean(loss)
    return probs, mean_loss
