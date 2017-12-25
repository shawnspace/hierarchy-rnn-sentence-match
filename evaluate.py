import tensorflow as tf
import model
import hparams as hp
import input
import os
import sys
import metrics

tf.flags.DEFINE_string('model_dir', None, 'The directory of model')
tf.flags.DEFINE_string('input_dir', './data', 'the dir of test.tfrecords')
tf.flags.DEFINE_integer('eval_batch_size', 10, 'the batch size of test')
tf.flags.DEFINE_integer("loglevel", 20, "Tensorflow log level")
FLAGS = tf.flags.FLAGS

if not FLAGS.model_dir:
  print("You must specify a model directory")
  sys.exit(1)

TEST_FILE_PATH = os.path.join(os.path.abspath(FLAGS.input_dir), 'test.tfrecords')

tf.logging.set_verbosity(FLAGS.loglevel)

def main(unused_arg):
    model_fn = model.create_model_fn(hp.create_hparams())

    estimator = tf.contrib.learn.Estimator(model_fn=model_fn,model_dir=FLAGS.model_dir,config=tf.contrib.learn.RunConfig())

    input_fn = input.create_input_fn([TEST_FILE_PATH],tf.crontrib.learn.ModeKeys.EVAL,FLAGS.test_batch_size,1)

    eval_metrics = metrics.create_evaluation_metrics()

    estimator.evaluate(input_fn=input_fn,batch_size=FLAGS.test_batch_size,metrics=eval_metrics,steps=None)

if __name__ == '__main__':
    tf.app.run()
