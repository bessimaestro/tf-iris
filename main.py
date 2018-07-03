import os
import glob
import shutil
import typing
import six.moves.urllib.request as request
import tensorflow as tf

path = os.getcwd()
file_train = os.path.join(path, 'dataset', 'iris-training.csv')
file_test = os.path.join(path, 'dataset', 'iris-test.csv')
url_train = 'http://download.tensorflow.org/data/iris_training.csv'
url_test = 'http://download.tensorflow.org/data/iris_test.csv'
logdir = os.path.join(path, 'logs')


def download_dataset(url: str, file: str):
    if not os.path.exists(os.path.join(path, 'dataset')):
        os.mkdir(os.path.join(path, 'dataset'))

    if not os.path.exists(file):
        data = request.urlopen(url).read()
        with open(file, 'wb') as f:
            f.write(data)


def input_fn(file_path: str, shuffle: bool=False, repeat_count=1):

    def decode_csv(line: str):
        parsed_line = tf.decode_csv(line, record_defaults=[[0.], [0.], [0.], [0.], [0]])
        label = parsed_line[-1]
        del parsed_line[-1]
        features = parsed_line
        d = dict(zip(feature_names, features)), label
        return d

    dataset = (tf.data.TextLineDataset(file_path)
               .skip(1)
               .map(decode_csv, num_parallel_calls=os.cpu_count()))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=256, seed=42)
    dataset = dataset.repeat(repeat_count)
    dataset = dataset.batch(32)
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


def predict_input_fn():

    def decode(x):
        x = tf.split(x, 4)
        return dict(zip(feature_names, x))

    dataset = tf.data.Dataset.from_tensor_slices(prediction_input)
    dataset = dataset.map(decode, num_parallel_calls=os.cpu_count())
    iterator = dataset.make_one_shot_iterator()
    next_feature_batch = iterator.get_next()
    return next_feature_batch, None


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
    feature_columns = [tf.feature_column.numeric_column(k) for k in feature_names]

    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[16, 16, 8],
                                            n_classes=3,
                                            model_dir=logdir)

    classifier.train(input_fn=lambda: input_fn(file_train, True, 32))
    evaluate_result = classifier.evaluate(input_fn=lambda: input_fn(file_test, False, 1))

    # print("Evaluation results:")
    for key, val in evaluate_result.items():
        print("{}: {}".format(key, val))

    predict_results = classifier.predict(input_fn=lambda: input_fn(file_test, False, 1))
    # print("Predictions on test file")
    for prediction in predict_results:
        print(prediction['class_ids'][0])

    prediction_input = [[5.9, 3.0, 4.2, 1.5],  # -> 1, Iris Versicolor
                        [6.9, 3.1, 5.4, 2.1],  # -> 2, Iris Virginica
                        [5.1, 3.3, 1.7, 0.5]]  # -> 0, Iris Setosa

    species = {0: 'Iris Setosa',
               1: 'Iris Versicolor',
               2: 'Iris Virginica'}

    predict_results = classifier.predict(input_fn=predict_input_fn)
    for i, prediction in enumerate(predict_results):
        pred = prediction['class_ids'][0]
        print("{}".format(species.get(pred)))