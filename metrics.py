import tensorflow as tf

def create_metric_spec(metric_fn, pre_key, label_key=None):
    return tf.contrib.learn.MetricSpec(metric_fn=metric_fn,prediction_key=pre_key,label_key=label_key)

def create_evaluation_metrics(dataset_name):
    metrics = {'{}_auc_value'.format(dataset_name): tf.contrib.learn.MetricSpec(metric_fn=tf.contrib.metrics.streaming_auc,prediction_key='probs'),'{}_accuracy'.format(dataset_name):tf.contrib.learn.MetricSpec(metric_fn=tf.contrib.metrics.streaming_accuracy,prediction_key='predictions')}
    return metrics
