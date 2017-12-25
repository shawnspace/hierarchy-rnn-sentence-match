
"""

train the model

"""

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import model
import input
import os
import time
import hparams as hp
import metrics

tf.flags.DEFINE_string("model_dir", None, 'the directory of model file')
tf.flags.DEFINE_string('input_dir', './data','directory contains train.tfrecords and validation.tfrecords')
tf.flags.DEFINE_integer("loglevel", tf.logging.WARN, "Tensorflow log level")
tf.flags.DEFINE_integer("eval_every", 105, "Evaluate after this many train steps")
tf.flags.DEFINE_integer('train_eval_every', 105, 'Evaluate training loss after this many steps')
tf.flags.DEFINE_boolean('debug', False, 'debug mode')

FLAGS = tf.flags.FLAGS

if FLAGS.model_dir is not None:
    MODEL_DIR = FLAGS.model_dir
else:
    TIMESTAMP = str(time.time())
    MODEL_DIR = os.path.join("./model", TIMESTAMP)

TRAIN_FILE_PATH = os.path.join(os.path.abspath(FLAGS.input_dir), 'train.tfrecords')
VALIDATION_FILE_PATH = os.path.join(os.path.abspath(FLAGS.input_dir), 'validation.tfrecords')
TEST_FILE_PATH = os.path.join(os.path.abspath(FLAGS.input_dir), 'test.tfrecords')

tf.logging.set_verbosity(FLAGS.loglevel)

def main(unused_arg):
    hparams = hp.create_hparams()

    model_fn = model.create_model_fn(hparams)

    estimator = tf.contrib.learn.Estimator(model_fn= model_fn, config=tf.contrib.learn.RunConfig(save_checkpoints_steps=FLAGS.eval_every,save_summary_steps=10000, log_step_count_steps=10000,model_dir=MODEL_DIR))

    input_fn_train = input.create_input_fn(input_files=[TRAIN_FILE_PATH], batch_size=hparams.batch_size,mode=tf.contrib.learn.ModeKeys.TRAIN,num_epochs=hparams.num_epochs)

    monitors_list = []

    input_fn_validation = input.create_input_fn([VALIDATION_FILE_PATH],tf.contrib.learn.ModeKeys.EVAL,hparams.eval_batch_size,1)
    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        input_fn=input_fn_validation,
        every_n_steps=FLAGS.eval_every,
        metrics=metrics.create_evaluation_metrics('validation'))
    monitors_list.append(validation_monitor)

    input_fn_test = input.create_input_fn([TEST_FILE_PATH],tf.contrib.learn.ModeKeys.EVAL,hparams.eval_batch_size,1)
    test_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        input_fn=input_fn_test,
        every_n_steps=FLAGS.eval_every,
        metrics=metrics.create_evaluation_metrics('test'))
    monitors_list.append(test_monitor)

    if FLAGS.debug:
        debuger = tf_debug.LocalCLIDebugHook()
        monitors_list.append(debuger)

    input_fn_train_eval = input.create_input_fn([TRAIN_FILE_PATH],tf.contrib.learn.ModeKeys.EVAL,hparams.batch_size,1)
    train_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        input_fn=input_fn_train_eval,
        every_n_steps=FLAGS.train_eval_every,
        metrics={'train_accuracy':metrics.create_metric_spec(tf.contrib.metrics.streaming_accuracy,'predictions',None)})
    monitors_list.append(train_monitor)

    estimator.fit(input_fn = input_fn_train, steps = None, monitors=monitors_list)

    hp.write_hparams_to_file(hparams, MODEL_DIR)

if __name__ == '__main__':
    tf.app.run()





