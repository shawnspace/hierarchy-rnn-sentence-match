"""
create input_fn to train, test, validation

"""

import tensorflow as tf

MAX_SENTENCE_LENGTH = 50
CONTEXT_WIDTH = 4

def create_features_spec():
    feature_spec = {}
    feature_spec['query'] = tf.FixedLenFeature(shape=[MAX_SENTENCE_LENGTH], dtype=tf.int64)
    feature_spec['query_len'] = tf.FixedLenFeature(shape=[1], dtype=tf.int64)
    feature_spec['label'] = tf.FixedLenFeature(shape=[1], dtype=tf.int64)

    for i in range(CONTEXT_WIDTH):
        feature_spec['context_{}'.format(i)] = tf.FixedLenFeature(shape=[MAX_SENTENCE_LENGTH], dtype=tf.int64)
        feature_spec['context_len_{}'.format(i)] = tf.FixedLenFeature(shape=[1], dtype=tf.int64)
    return feature_spec

def create_input_fn(input_files: object, mode: object, batch_size: object, num_epochs: object) -> object:

    def input_fn():

        features_spec = create_features_spec()
        feature_map = tf.contrib.learn.io.read_batch_features(file_pattern=input_files, batch_size=batch_size,
                                                              features=features_spec, reader=tf.TFRecordReader,
                                                              num_epochs=num_epochs, randomize_input=True,
                                                              name="read_batch_features_{}".format(mode))
        target = feature_map.pop('label')
        target = tf.squeeze(target,axis=[1],name='labels')

        return feature_map, target
    return input_fn



