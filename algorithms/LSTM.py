import pandas as pd
from keras import backend as K
from keras import optimizers
from keras.layers import Dense, Dropout, Input
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import Model
from keras.models import load_model
from keras.callbacks import *

from algorithms.data_util import DataLoader, pad, plot_loss_and_accuracy, pad_data_list
from utils.tokenizer_util import HebrewTokenizer

MODEL_PATH = '..\models\LSTM-Token-88.242.h5'

# Preprocess Data
# load
all_data = DataLoader('../data/token_train.tsv', '../data/token_test.tsv', '../data/edge')

# tokenize
heb_tokenizer = HebrewTokenizer(all_data.x_token_train, init_from_file=True)
x_token_train = heb_tokenizer.texts_to_sequences(all_data.x_token_train)
x_token_test = heb_tokenizer.texts_to_sequences(all_data.x_token_test)

x_tweets_dict_pre_processed = {}
for key, x_tweets in all_data.x_tweets_dict.items():
    x_tweets_pre_processed = heb_tokenizer.texts_to_sequences(list(x_tweets['full_text']))
    x_tweets_pre_processed = pad_data_list([list(x_tweets_pre_processed)])[0]
    x_tweets_dict_pre_processed[key] = x_tweets_pre_processed

y_token_test = all_data.y_token_test
y_token_train = all_data.y_token_train

# pad
x_token_train, x_token_test = pad(x_token_train, x_token_test)
# x_tweets = pad_data_list([x_tweets])[0]


def train_model():
    # Default Parameters
    max_document_length = 100
    dropout_keep_prob = 0.5
    embedding_size = 300
    batch_size = 50
    lr = 1e-4
    dev_size = 0.2
    vocabulary_size = 5000

    ## CNN - Token
    num_epochs = 5
    ## LSTM - Token
    # %%
    num_epochs = 100
    lstm_units = 93

    # Create new TF graph
    K.clear_session()

    # Construct model
    text_input = Input(shape=(max_document_length,))
    x = Embedding(vocabulary_size, embedding_size)(text_input)
    x = LSTM(units=lstm_units, return_sequences=True, activation='relu')(x)
    x = LSTM(units=lstm_units)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_keep_prob)(x)
    preds = Dense(3, activation='softmax')(x)

    model = Model(text_input, preds)

    adam = optimizers.Adam(lr=lr)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    # Train the model
    es = EarlyStopping(patience=3)
    rlrop = ReduceLROnPlateau(patience=2)
    history = model.fit(x_token_train, y_token_train,
                        batch_size=batch_size,
                        epochs=num_epochs,
                        verbose=1,
                        validation_split=dev_size, callbacks=[es, rlrop])

    # Plot training accuracy and loss
    plot_loss_and_accuracy(history)

    # Evaluate the model
    scores = model.evaluate(x_token_test, y_token_test,
                            batch_size=batch_size, verbose=1)
    print('\nAccurancy: {:.3f}'.format(scores[1]))

    # Save the model
    model.save('../models/LSTM-Token-{:.3f}.h5'.format((scores[1] * 100)))


def predict_tweets():
    new_model = load_model(MODEL_PATH)
    for file_name, x_tweets_processed in x_tweets_dict_pre_processed.items():
        predictions = new_model.predict(x_tweets_processed)
        predictions_prob = np.max(predictions, axis=1)
        predictions_binary = predictions.argmax(axis=1)
    # create predictions file

        predictions_binary = pd.DataFrame(predictions_binary)
        predictions_prob = pd.DataFrame(predictions_prob)

        tweets = pd.DataFrame(all_data.x_tweets_dict[file_name])
        tweets['preds_binary'] = predictions_binary[0]
        tweets['preds_prob'] = predictions_prob[0]
        # os.path.splitext("path_to_file")[0]
        # tweets.to_csv('..\\results\\' + os.path.splitext(file_name)[0] + '.csv', encoding='utf-8', index=False, header=False)
        # tweets.to_csv('..\\results\\' + file_name, encoding='utf-8', index=False, header=False)
        tweets.to_csv('..\\results\\edge\\lstm\\' + file_name, header=True, encoding='utf-8')
    print("Done prediction")


# train_model()
predict_tweets()
