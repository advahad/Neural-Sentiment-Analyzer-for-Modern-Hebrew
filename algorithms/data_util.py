import codecs

import numpy as np
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical

import matplotlib.pyplot as plt


def plot_loss_and_accuracy(history):
    fig, axs = plt.subplots(1, 2, sharex=True)

    axs[0].plot(history.history['loss'])
    axs[0].plot(history.history['val_loss'])
    axs[0].set_title('Model Loss')
    axs[0].legend(['Train', 'Validation'], loc='upper left')

    axs[1].plot(history.history['acc'])
    axs[1].plot(history.history['val_acc'])
    axs[1].set_title('Model Accuracy')
    axs[1].legend(['Train', 'Validation'], loc='upper left')

    fig.tight_layout()
    plt.show()


def load_data(filename):
    data = list(codecs.open(filename, 'r', 'utf-8').readlines())
    x, y = zip(*[d.strip().split('\t') for d in data])
    # Reducing any char-acter sequence of more than 3 consecutive repetitions to a respective 3-character sequence
    # (e.g. “!!!!!!!!”turns to “!!!”)
    # x = [re.sub(r'((.)\2{3,})', r'\2\2\2', i) for i in x]
    x = np.asarray(list(x))
    y = to_categorical(y, 3)

    return x, y


def load_data_tweets(filename):
    data = list(codecs.open(filename, 'r', 'utf-8').readlines())
    # x= zip(*[d.strip() for d in data])
    # Reducing any char-acter sequence of more than 3 consecutive repetitions to a respective 3-character sequence
    # (e.g. “!!!!!!!!”turns to “!!!”)
    # x = [re.sub(r'((.)\2{3,})', r'\2\2\2', i) for i in x]
    x = np.asarray(list(data))

    return x


def pad_data_list(data_list, max_document_length=100):
    padded_data_list = []
    for data_set in data_list:
        padded_data_list.append(sequence.pad_sequences(data_set, maxlen=max_document_length, padding='post',
                                                       truncating='post'))
    return padded_data_list


def pad(x_train, x_test, max_document_length=100):
    x_train = sequence.pad_sequences(x_train, maxlen=max_document_length, padding='post', truncating='post')
    x_test = sequence.pad_sequences(x_test, maxlen=max_document_length, padding='post', truncating='post')

    return x_train, x_test


class DataLoader:
    def __init__(self, train_file_name, test_file_name, tweets_file_name):
        self.x_token_train, self.y_token_train = load_data(train_file_name)
        self.x_token_test, self.y_token_test = load_data(test_file_name)
        self.x_tweets_test = load_data_tweets(tweets_file_name)

        # print('X token train shape: {}'.format(x_token_train.shape))
        # print('X token test shape: {}'.format(x_token_test.shape))