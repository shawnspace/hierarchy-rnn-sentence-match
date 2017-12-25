"""
create model_fn to train.py

"""

import tensorflow as tf
import hierarchy_rnn

MAX_SENTENCE_LENGTH = 50
MAX_CONTEXT_WIDTH = 4

def get_feature(features, key, len_key, max_sen_len):
    feature = features[key]
    feature_len = tf.squeeze(features[len_key], axis=[1]) #[batch_size, 1] -> [batch_size]
    feature_len = tf.minimum(feature_len, tf.constant(MAX_SENTENCE_LENGTH, dtype=tf.int64))
    return feature, feature_len

def get_train_op(hparams, loss):
    return tf.contrib.layers.optimize_loss(loss=loss,global_step=tf.contrib.framework.get_global_step(),learning_rate=hparams.learning_rate,optimizer=hparams.optimizer)

def create_model_fn(hparams):
    def model_fn(features, labels, mode):

        contexts = []
        contexts_len = []
        for i in range(MAX_CONTEXT_WIDTH):
            context, context_len = get_feature(features, 'context_{}'.format(i), 'context_len_{}'.format(i), MAX_SENTENCE_LENGTH)
            contexts.append(context)
            contexts_len.append(context_len)
        query, query_len = get_feature(features, 'query', 'query_len', MAX_SENTENCE_LENGTH)

        if mode == tf.contrib.learn.ModeKeys.TRAIN:
            prob, loss = hierarchy_rnn.hierarchy_rnn(hparams,mode=tf.contrib.learn.ModeKeys.TRAIN,contexts=contexts,contexts_len=contexts_len,query=query, query_len=query_len, label=labels)
            train_op = get_train_op(hparams,loss)
            prediction = tf.round(prob,'prob_to_prediction')
            return {'probs':prob, 'predictions':prediction}, loss, train_op

        if mode == tf.contrib.learn.ModeKeys.INFER:
            prob, loss = hierarchy_rnn.hierarchy_rnn(hparams,mode=tf.contrib.learn.ModeKeys.INFER,contexts=contexts,contexts_len=contexts_len,query=query, query_len=query_len, label=None)
            prediction = tf.round(prob)
            return prediction, 0.0, None

        if mode == tf.contrib.learn.ModeKeys.EVAL:
            prob, loss = hierarchy_rnn.hierarchy_rnn(hparams,mode=tf.contrib.learn.ModeKeys.EVAL,contexts=contexts,contexts_len=contexts_len,query=query, query_len=query_len, label=labels)
            prediction = tf.round(prob)
            return {'probs':prob,'predictions':prediction}, loss, None

    return model_fn