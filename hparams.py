import tensorflow as tf
from collections import namedtuple
import os

# Model Parameters
tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of the embeddings")
tf.flags.DEFINE_integer('sentence_level_encoding_num_units', 64, 'Dim of sentence encoding vector')
tf.flags.DEFINE_integer('session_level_encoding_num_units', 64, 'Dim of context encoding vector')
tf.flags.DEFINE_float('keep_prob', 0.5, 'the keep prob of rnn state')
tf.flags.DEFINE_string('rnn_cell_type', 'BasicLSTM', 'the cell type in rnn')

# Pre-trained embeddings
tf.flags.DEFINE_string("glove_path", './data/word_embed_50.txt', "Path to pre-trained Glove vectors")
tf.flags.DEFINE_string("vocab_path", './data/vocabulary.txt', "Path to vocabulary.txt file")

# Training Parameters
tf.flags.DEFINE_float("learning_rate", 0.0001, "Learning rate")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size during training")
tf.flags.DEFINE_integer("eval_batch_size", 1, "Batch size during evaluation")
tf.flags.DEFINE_string("optimizer", "Adam", "Optimizer Name (Adam, Adagrad, etc)")
tf.flags.DEFINE_integer('num_epochs', 30, 'the number of epochs')

FLAGS = tf.flags.FLAGS

HParams = namedtuple(
  "HParams",
  [
    "batch_size",
    "embedding_dim",
    "eval_batch_size",
    "learning_rate",
    "optimizer",
    "glove_path",
    "vocab_path",
    "num_epochs",
    'sentence_level_encoding_num_units',
    'session_level_encoding_num_units',
    'keep_prob',
    'rnn_cell_type'
  ])

def create_hparams():
  return HParams(
    batch_size=FLAGS.batch_size,
    eval_batch_size=FLAGS.eval_batch_size,
    optimizer=FLAGS.optimizer,
    learning_rate=FLAGS.learning_rate,
    embedding_dim=FLAGS.embedding_dim,
    glove_path=FLAGS.glove_path,
    vocab_path=FLAGS.vocab_path,
    num_epochs=FLAGS.num_epochs,
    sentence_level_encoding_num_units=FLAGS.sentence_level_encoding_num_units,
    session_level_encoding_num_units=FLAGS.session_level_encoding_num_units,
    keep_prob=FLAGS.keep_prob,
    rnn_cell_type=FLAGS.rnn_cell_type
  )

def write_hparams_to_file(hp, model_dir):
  with open(os.path.join(os.path.abspath(model_dir),'hyper_parameters.txt'), 'w') as f:
    f.write('batch_size: {}\n'.format(hp.batch_size))
    f.write('learning_rate: {}\n'.format(hp.learning_rate))
    f.write('num_epochs: {}\n'.format(hp.num_epochs))
    f.write('sentence_level_encoding_num_units: {}\n'.format(hp.sentence_level_encoding_num_units))
    f.write('session_level_encoding_num_units: {}\n'.format(hp.session_level_encoding_num_units))
    f.write('keep_prob: {}\n'.format(hp.keep_prob))
    f.write('rnn_cell_type: {}\n'.format(hp.rnn_cell_type))
    f.write('optimizer: {}\n'.format(hp.optimizer))
