import pandas as pd
from keras import backend as K
from keras import optimizers
from keras.layers import Dense, Dropout, Input
from keras.layers import Embedding
from keras.layers import LSTM, Bidirectional
# %% md
## Import required modules from Keras
# %%
from keras.models import Model
from keras.models import load_model

from algorithms.data_util import DataLoader, pad, plot_loss_and_accuracy, pad_data_list
from utils.tokenizer_util import HebrewTokenizer

MODEL_PATH = '..\models\Linear-Token-70.078.h5'



# Preprocess Data
# load
all_data = DataLoader('../data/token_train.tsv', '../data/token_test.tsv', '../data/tweets_netanyahu.txt')


# tokenize
heb_tokenizer = HebrewTokenizer(all_data.x_token_train, init_from_file=True)
x_token_train = heb_tokenizer.texts_to_sequences(all_data.x_token_train)
x_token_test = heb_tokenizer.texts_to_sequences(all_data.x_token_test)
x_tweets = heb_tokenizer.texts_to_sequences(all_data.x_tweets_dict)

y_token_test = all_data.y_token_test
y_token_train = all_data.y_token_train

# pad
x_token_train, x_token_test = pad(x_token_train, x_token_test)
x_tweets = pad_data_list([x_tweets])[0]


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
    num_epochs = 7
    lstm_units = 93

    num_epochs = 3

    # Create new TF graph
    K.clear_session()

    # Construct model
    text_input = Input(shape=(max_document_length,))
    x = Embedding(vocabulary_size, embedding_size)(text_input)
    x = Bidirectional(LSTM(units=lstm_units, return_sequences=True))(x)
    x = Bidirectional(LSTM(units=lstm_units))(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout_keep_prob)(x)
    preds = Dense(3, activation='softmax')(x)

    model = Model(text_input, preds)

    adam = optimizers.Adam(lr=lr)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(x_token_train, y_token_train,
                        batch_size=batch_size,
                        epochs=num_epochs,
                        verbose=1,
                        validation_split=dev_size)

    # Plot training accuracy and loss
    plot_loss_and_accuracy(history)

    # Evaluate the model
    scores = model.evaluate(x_token_test, y_token_test,
                            batch_size=batch_size, verbose=1)
    print('\nAccurancy: {:.3f}'.format(scores[1]))

    # Save the model
    model.save('../models/BiLSTM-Token-{:.3f}.h5'.format((scores[1] * 100)))

def predict_tweets():
    new_model = load_model(MODEL_PATH)
    predictions = new_model.predict(x_tweets)
    predictions = predictions.argmax(axis=1)
    # create predictions file
    predictions = pd.DataFrame(predictions)
    tweets = pd.DataFrame(all_data.x_tweets_dict)
    tweets['preds'] = predictions[0]
    tweets.to_csv('..\\results\\labeled_tweets.csv', encoding='utf-8', index=False, header=False)


train_model()
