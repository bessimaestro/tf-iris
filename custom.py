import os
import glob
import shutil
from typing import Tuple, Dict, List
import six.moves.urllib.request as request
import tensorflow as tf

path = os.getcwd()
file_train = os.path.join(path, 'dataset', 'iris-training.csv')
file_test = os.path.join(path, 'dataset', 'iris-test.csv')
url_train = 'http://download.tensorflow.org/data/iris_training.csv'
url_test = 'http://download.tensorflow.org/data/iris_test.csv'
logdir = os.path.join(path, 'logs')


def download_dataset(url: str, file: str) -> None:
    if not os.path.exists(os.path.join(path, 'dataset')):
        os.mkdir(os.path.join(path, 'dataset'))

    if not os.path.exists(file):
        data = request.urlopen(url).read()
        with open(file, 'wb') as f:
            f.write(data)


def input_fn(file_path: str,
             repeat_count: int = 1,
             shuffle_count: int = 1) -> Tuple:

    def decode_csv(line: str) -> Tuple:
        parsed_line = tf.decode_csv(line, record_defaults=[[0.], [0.], [0.], [0.], [0]])
        label = parsed_line[-1]
        del parsed_line[-1]
        features = parsed_line
        d = dict(zip(feature_names, features)), label
        return d

    dataset = (tf.data.TextLineDataset(file_path)
        .skip(1)
        .map(decode_csv, num_parallel_calls=os.cpu_count())
        .shuffle(buffer_size=shuffle_count, seed=42)
        .repeat(repeat_count)
        .batch(32)
        .prefetch(1))

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


def model_fn(features: Dict,
             labels: List,
             mode: tf.estimator.ModeKeys) -> tf.estimator.EstimatorSpec:

    if mode == tf.estimator.ModeKeys.PREDICT:
        tf.logging.info("model_fn: PREDICT, {}".format(mode))
    elif mode == tf.estimator.ModeKeys.EVAL:
            tf.logging.info("model_fn: EVAL, {}".format(mode))
    elif mode == tf.estimator.ModeKeys.TRAIN:
            tf.logging.info("model_fn: TRAIN, {}".format(mode))

    input_layer = tf.feature_column.input_layer(features, feature_columns)
    h1 = tf.layers.Dense(10, activation=tf.nn.relu)(input_layer)
    h2 = tf.layers.Dense(10, activation=tf.nn.relu)(h1)
    logits = tf.layers.Dense(3)(h2)
    predictions = {'class_ids': tf.argmax(input=logits, axis=1)}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.metrics.accuracy(labels, predictions['class_ids'])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode,
                                          loss=loss,
                                          eval_metric_ops={'accuracy': accuracy})

    assert mode == tf.estimator.ModeKeys.TRAIN, "TRAIN is only ModeKey left"

    optimizer = tf.train.GradientDescentOptimizer(0.05)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    tf.summary.scalar('accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(mode,
                                      loss=loss,
                                      train_op=train_op)


if __name__ == '__main__':

    if os.path.exists(logdir):
        for f in glob.glob(os.path.join(logdir, '*')):
            if os.path.isdir(f):
                shutil.rmtree(f)
            else:
                os.remove(f)
    else:
        os.mkdir(logdir)

    download_dataset(url_train, file_train)
    download_dataset(url_test, file_test)

    tf.logging.set_verbosity(tf.logging.INFO)

    feature_names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
    feature_columns = [tf.feature_column.numeric_column(feat) for feat in feature_names]

    tf.logging.info("Before classifier construction")
    classifier = tf.estimator.Estimator(model_fn=model_fn,
                                        model_dir=logdir)
    tf.logging.info("...done constructing classifier")

    tf.logging.info("Before classifier.train")
    classifier.train(input_fn=lambda: input_fn(file_train, 500, 256))
    tf.logging.info("...done classifier.train")

    tf.logging.info("Before classifier.evaluate")
    evaluate_result = classifier.evaluate(input_fn=lambda: input_fn(file_test, 4))
    tf.logging.info("...done classifier.evauate")
    for key, value in evaluate_result.items():
        tf.logging.info("{} {}".format(key, value))

    tf.logging.info("Before classifier.predict")
    predict_result = classifier.predict(input_fn=lambda: input_fn(file_test, 4))
    tf.logging.info("...done classifier.predict")
    for prediction in predict_result:
        tf.logging.info("...{}".format(prediction['class_ids']))

