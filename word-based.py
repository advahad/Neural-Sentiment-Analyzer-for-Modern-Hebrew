## Load data files
import codecs
from keras.utils.np_utils import to_categorical
import numpy as np
from keras.models import load_model

import pickle


def load_data(filename):
    data = list(codecs.open(filename, 'r', 'utf-8').readlines())
    x, y = zip(*[d.strip().split('\t') for d in data])
    # Reducing any char-acter sequence of more than 3 consecutive repetitions to a respective 3-character sequence 
    # (e.g. “!!!!!!!!”turns to “!!!”)
    # x = [re.sub(r'((.)\2{3,})', r'\2\2\2', i) for i in x]
    x = np.asarray(list(x))
    y = to_categorical(y, 3)

    return x, y


def load_data_twitt(filename):
    data = list(codecs.open(filename, 'r', 'utf-8').readlines())
    # x= zip(*[d.strip() for d in data])
    # Reducing any char-acter sequence of more than 3 consecutive repetitions to a respective 3-character sequence
    # (e.g. “!!!!!!!!”turns to “!!!”)
    # x = [re.sub(r'((.)\2{3,})', r'\2\2\2', i) for i in x]
    x = np.asarray(list(data))

    return x


x_token_train, y_token_train = load_data('data/token_train.tsv')
x_token_test, y_token_test = load_data('data/token_test.tsv')
x_morph_train, y_morph_train = load_data('data/morph_train.tsv')
x_morph_test, y_morph_test = load_data('data/morph_test.tsv')
# x_twitts_test = load_data_twitt('data/tweets_netanyahu.txt')


print('X token test shape: {}'.format(x_token_test.shape))

print('X token train shape: {}'.format(x_token_train.shape))

# print('X token twitts test shape: {}'.format(x_twitts_test.shape))

print('X morph train shape: {}'.format(x_morph_train.shape))
print('X morph test shape: {}'.format(x_morph_test.shape))

# print(x_token_train[:5])


# print(x_token_test[:5])


#
# print(x_morph_train[:5])
#
#
# # In[86]:
#
#
# print(x_morph_test[:5])


# ## Prepare
# Convert text (train & test) to sequences and pad to requested document length



from keras.preprocessing import text, sequence


def tokenizer(x_train, x_test, vocabulary_size, char_level):
    tokenize = text.Tokenizer(num_words=vocabulary_size,
                              char_level=char_level,
                              filters='')
    tokenize.fit_on_texts(x_train)  # only fit on train
    # print('UNK index: {}'.format(tokenize.word_index['UNK']))

    x_train = tokenize.texts_to_sequences(x_train)
    x_test = tokenize.texts_to_sequences(x_test)

    return x_train, x_test

def pad(x_train, x_test, max_document_length):
    x_train = sequence.pad_sequences(x_train, maxlen=max_document_length, padding='post', truncating='post')
    x_test = sequence.pad_sequences(x_test, maxlen=max_document_length, padding='post', truncating='post')

    return x_train, x_test


vocabulary_size = 5000

x_token_train, x_token_test = tokenizer(x_token_train, x_token_test, vocabulary_size, False)
x_morph_train, x_morph_test = tokenizer(x_morph_train, x_morph_test, vocabulary_size, False)
# x_token_train_redundant, x_twitts_test = tokenizer(x_token_train, x_twitts_test, vocabulary_size, False)

max_document_length = 100

x_token_train, x_token_test = pad(x_token_train, x_token_test, max_document_length)
x_morph_train, x_morph_test = pad(x_morph_train, x_morph_test, max_document_length)
# x_twitts_test, x_twitts_test_redandent = pad(x_twitts_test, x_twitts_test, max_document_length)

# print('X token twitts test shape: {}'.format(x_twitts_test.shape))

print('X token train shape: {}'.format(x_token_train.shape))
print('X token test shape: {}'.format(x_token_test.shape))

print('X morph train shape: {}'.format(x_morph_train.shape))
print('X morph test shape: {}'.format(x_morph_test.shape))

# In[88]:


print('Token OOV ratio: {} ({} out of 28787)'.format(np.count_nonzero(x_token_test == 28787) / 28787,
                                                     np.count_nonzero(x_token_test == 28787)))
print('Morph OOV ratio: {} ({} out of 18912)'.format(np.count_nonzero(x_morph_test == 18912) / 18912,
                                                     np.count_nonzero(x_morph_test == 18912)))

# ## Plot function

# In[5]:


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


# ## Import required modules from Keras

# In[6]:


from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Concatenate
from keras.layers import Embedding
from keras.layers import LSTM, Bidirectional
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D
from keras.layers import BatchNormalization
from keras import optimizers
from keras import metrics
from keras import backend as K

# ## Default Parameters

# In[7]:


dropout_keep_prob = 0.5
embedding_size = 300
batch_size = 50
lr = 1e-4
dev_size = 0.2

# ## Linear - Token

# In[8]:


num_epochs = 100

# Create new TF graph
K.clear_session()

# Construct model
text_input = Input(shape=(max_document_length,))
x = Dense(100)(text_input)
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
print('\nAccurancy: {:.4f}'.format(scores[1]))

# Save the model
model.save('models\Linear-Token-{:.3f}.h5'.format((scores[1] * 100)))


# new_model = load_model('Linear-Token-70.117.h5')
# predictions = new_model.predict(x_twitts_test)
# predictions = predictions.argmax(axis=1)
# print(predictions)

# ## CNN - Token

# In[105]:

#
# num_epochs = 5
#
# # Create new TF graph
# K.clear_session()
#
# # Construct model
# convs = []
# text_input = Input(shape=(max_document_length,))
# x = Embedding(vocabulary_size, embedding_size)(text_input)
# for fsz in [3, 8]:
#     conv = Conv1D(128, fsz, padding='valid', activation='relu')(x)
#     pool = MaxPool1D()(conv)
#     convs.append(pool)
# x = Concatenate(axis=1)(convs)
# x = Flatten()(x)
# x = Dense(128, activation='relu')(x)
# x = Dropout(dropout_keep_prob)(x)
# preds = Dense(3, activation='softmax')(x)
#
# model = Model(text_input, preds)
#
# adam = optimizers.Adam(lr=lr)
#
# model.compile(loss='categorical_crossentropy',
#               optimizer=adam,
#               metrics=['accuracy'])
#
# # Train the model
# history = model.fit(x_token_train, y_token_train,
#                     batch_size=batch_size,
#                     epochs=num_epochs,
#                     verbose=1,
#                     validation_split=dev_size)
#
# # Plot training accuracy and loss
# plot_loss_and_accuracy(history)
#
# # Evaluate the model
# scores = model.evaluate(x_token_test, y_token_test,
#                         batch_size=batch_size, verbose=1)
# print('\nAccurancy: {:.3f}'.format(scores[1]))
#
# # Save the model
# model.save('word_saved_models/CNN-Token-{:.3f}.h5'.format((scores[1] * 100)))
#
#
# # ## CNN - Morph
#
# # In[106]:
#
#
# # Create new TF graph
# K.clear_session()
#
# # Construct model
# convs = []
# text_input = Input(shape=(max_document_length,))
# x = Embedding(vocabulary_size, embedding_size)(text_input)
# for fsz in [3, 8]:
#     conv = Conv1D(128, fsz, padding='valid', activation='relu')(x)
#     pool = MaxPool1D()(conv)
#     convs.append(pool)
# x = Concatenate(axis=1)(convs)
# x = Flatten()(x)
# x = Dense(128, activation='relu')(x)
# x = Dropout(dropout_keep_prob)(x)
# preds = Dense(3, activation='softmax')(x)
#
# model = Model(text_input, preds)
#
# adam = optimizers.Adam(lr=lr)
#
# model.compile(loss='categorical_crossentropy',
#               optimizer=adam,
#               metrics=['accuracy'])
#
# # Train the model
# history = model.fit(x_morph_train, y_morph_train,
#                     batch_size=batch_size,
#                     epochs=num_epochs,
#                     verbose=1,
#                     validation_split=dev_size)
#
# # Plot training accuracy and loss
# plot_loss_and_accuracy(history)
#
# # Evaluate the model
# scores = model.evaluate(x_morph_test, y_morph_test,
#                        batch_size=batch_size, verbose=1)
# print('\nAccurancy: {:.4f}'.format(scores[1]))
#
# # Save the model
# model.save('word_saved_models/CNN-Morph-{:.3f}.h5'.format((scores[1] * 100)))
#
#
# # ## LSTM - Token
#
# # In[114]:
#
#
# num_epochs = 7
# lstm_units = 93
#
# # Create new TF graph
# K.clear_session()
#
# # Construct model
# text_input = Input(shape=(max_document_length,))
# x = Embedding(vocabulary_size, embedding_size)(text_input)
# x = LSTM(units=lstm_units, return_sequences=True)(x)
# x = LSTM(units=lstm_units)(x)
# x = Dense(128, activation='relu')(x)
# x = Dropout(dropout_keep_prob)(x)
# preds = Dense(3, activation='softmax')(x)
#
# model = Model(text_input, preds)
#
# adam = optimizers.Adam(lr=lr)
#
# model.compile(loss='categorical_crossentropy',
#               optimizer=adam,
#               metrics=['accuracy'])
#
# # Train the model
# history = model.fit(x_token_train, y_token_train,
#                     batch_size=batch_size,
#                     epochs=num_epochs,
#                     verbose=1,
#                     validation_split=dev_size)
#
# # Plot training accuracy and loss
# plot_loss_and_accuracy(history)
#
# # Evaluate the model
# scores = model.evaluate(x_token_test, y_token_test,
#                        batch_size=batch_size, verbose=1)
# print('\nAccurancy: {:.3f}'.format(scores[1]))
#
# # Save the model
# model.save('word_saved_models/LSTM-Token-{:.3f}.h5'.format((scores[1] * 100)))
#
#
# # ## LSTM - Morph
#
# # In[108]:
#
#
# num_epochs = 7
#
# # Create new TF graph
# K.clear_session()
#
# # Construct model
# text_input = Input(shape=(max_document_length,))
# x = Embedding(vocabulary_size, embedding_size)(text_input)
# x = LSTM(units=lstm_units, return_sequences=True)(x)
# x = LSTM(units=lstm_units)(x)
# x = Dense(128, activation='relu')(x)
# x = Dropout(dropout_keep_prob)(x)
# preds = Dense(3, activation='softmax')(x)
#
# model = Model(text_input, preds)
#
# adam = optimizers.Adam(lr=lr)
#
# model.compile(loss='categorical_crossentropy',
#               optimizer=adam,
#               metrics=['accuracy'])
#
# # Train the model
# history = model.fit(x_morph_train, y_morph_train,
#                     batch_size=batch_size,
#                     epochs=num_epochs,
#                     verbose=1,
#                     validation_split=dev_size)
#
# # Plot training accuracy and loss
# plot_loss_and_accuracy(history)
#
# # Evaluate the model
# scores = model.evaluate(x_morph_test, y_morph_test,
#                        batch_size=batch_size, verbose=1)
# print('\nAccurancy: {:.3f}'.format(scores[1]))
#
# # Save the model
# model.save('word_saved_models/LSTM-Morph-{:.3f}.h5'.format((scores[1] * 100)))
#
#
# # ## BiLSTM - Token
#
# # In[109]:
#
#
# num_epochs = 3
#
# # Create new TF graph
# K.clear_session()
#
# # Construct model
# text_input = Input(shape=(max_document_length,))
# x = Embedding(vocabulary_size, embedding_size)(text_input)
# x = Bidirectional(LSTM(units=lstm_units, return_sequences=True))(x)
# x = Bidirectional(LSTM(units=lstm_units))(x)
# x = Dense(128, activation='relu')(x)
# x = Dropout(dropout_keep_prob)(x)
# preds = Dense(3, activation='softmax')(x)
#
# model = Model(text_input, preds)
#
# adam = optimizers.Adam(lr=lr)
#
# model.compile(loss='categorical_crossentropy',
#               optimizer=adam,
#               metrics=['accuracy'])
#
# # Train the model
# history = model.fit(x_token_train, y_token_train,
#                     batch_size=batch_size,
#                     epochs=num_epochs,
#                     verbose=1,
#                     validation_split=dev_size)
#
# # Plot training accuracy and loss
# plot_loss_and_accuracy(history)
#
# # Evaluate the model
# scores = model.evaluate(x_token_test, y_token_test,
#                        batch_size=batch_size, verbose=1)
# print('\nAccurancy: {:.3f}'.format(scores[1]))
#
# # Save the model
# model.save('word_saved_models/BiLSTM-Token-{:.3f}.h5'.format((scores[1] * 100)))
#
#
# # ## BiLSTM - Morph
#
# # In[110]:
#
#
# # Create new TF graph
# K.clear_session()
#
# # Construct model
# text_input = Input(shape=(max_document_length,))
# x = Embedding(vocabulary_size, embedding_size)(text_input)
# x = Bidirectional(LSTM(units=lstm_units, return_sequences=True))(x)
# x = Bidirectional(LSTM(units=lstm_units))(x)
# x = Dense(128, activation='relu')(x)
# x = Dropout(dropout_keep_prob)(x)
# preds = Dense(3, activation='softmax')(x)
#
# model = Model(text_input, preds)
#
# adam = optimizers.Adam(lr=lr)
#
# model.compile(loss='categorical_crossentropy',
#               optimizer=adam,
#               metrics=['accuracy'])
#
# # Train the model
# history = model.fit(x_morph_train, y_morph_train,
#                     batch_size=batch_size,
#                     epochs=num_epochs,
#                     verbose=1,
#                     validation_split=dev_size)
#
# # Plot training accuracy and loss
# plot_loss_and_accuracy(history)
#
# # Evaluate the model
# scores = model.evaluate(x_morph_test, y_morph_test,
#                        batch_size=batch_size, verbose=1)
# print('\nAccurancy: {:.3f}'.format(scores[1]))
#
# # Save the model
# model.save('word_saved_models/BiLSTM-Morph-{:.3f}.h5'.format((scores[1] * 100)))
#
#
# # ## MLP - Token
#
# # In[111]:
#
#
# num_epochs = 6
#
# # Create new TF graph
# K.clear_session()
#
# # Construct model
# text_input = Input(shape=(max_document_length,))
# x = Embedding(vocabulary_size, embedding_size)(text_input)
# x = Flatten()(x)
# x = Dense(256, activation='relu')(x)
# x = Dropout(dropout_keep_prob)(x)
# x = Dense(128, activation='relu')(x)
# x = Dropout(dropout_keep_prob)(x)
# x = Dense(64, activation='relu')(x)
# x = Dropout(dropout_keep_prob)(x)
# preds = Dense(3, activation='softmax')(x)
#
# model = Model(text_input, preds)
#
# adam = optimizers.Adam(lr=lr)
#
# model.compile(loss='categorical_crossentropy',
#               optimizer=adam,
#               metrics=['accuracy'])
#
# # Train the model
# history = model.fit(x_token_train, y_token_train,
#                     batch_size=batch_size,
#                     epochs=num_epochs,
#                     verbose=1,
#                     validation_split=dev_size)
#
# # Plot training accuracy and loss
# plot_loss_and_accuracy(history)
#
# # Evaluate the model
# scores = model.evaluate(x_token_test, y_token_test,
#                        batch_size=batch_size, verbose=1)
# print('\nAccurancy: {:.3f}'.format(scores[1]))
#
# # Save the model
# model.save('word_saved_models/MLP-Token-{:.3f}.h5'.format((scores[1] * 100)))
#
#
# # ## MLP - Morph
#
# # In[112]:
#
#
# # Create new TF graph
# K.clear_session()
#
# # Construct model
# text_input = Input(shape=(max_document_length,))
# x = Embedding(vocabulary_size, embedding_size)(text_input)
# x = Flatten()(x)
# x = Dense(256, activation='relu')(x)
# x = Dropout(dropout_keep_prob)(x)
# x = Dense(128, activation='relu')(x)
# x = Dropout(dropout_keep_prob)(x)
# x = Dense(64, activation='relu')(x)
# x = Dropout(dropout_keep_prob)(x)
# preds = Dense(3, activation='softmax')(x)
#
# model = Model(text_input, preds)
#
# adam = optimizers.Adam(lr=lr)
#
# model.compile(loss='categorical_crossentropy',
#               optimizer=adam,
#               metrics=['accuracy'])
#
# # Train the model
# history = model.fit(x_morph_train, y_morph_train,
#                     batch_size=batch_size,
#                     epochs=num_epochs,
#                     verbose=1,
#                     validation_split=dev_size)
#
# # Plot training accuracy and loss
# plot_loss_and_accuracy(history)
#
# # Evaluate the model
# scores = model.evaluate(x_morph_test, y_morph_test,
#                        batch_size=batch_size, verbose=1)
# print('\nAccurancy: {:.3f}'.format(scores[1]))
#
# # Save the model
# model.save('word_saved_models/MLP-Morph-{:.3f}.h5'.format((scores[1] * 100)))


# In[ ]:
