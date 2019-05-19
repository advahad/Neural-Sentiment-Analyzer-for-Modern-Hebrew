from algorithms.data_util import DataLoader, pad
from utils.tokenizer_util import HebrewTokenizer

# Preprocess Data

# load
model_data = DataLoader('data/token_train.tsv', 'data/token_test.tsv')

# tokenize
heb_tokenizer = HebrewTokenizer(model_data.x_token_train, init_from_file=True)
x_token_train = heb_tokenizer.texts_to_sequences(model_data.x_token_train)
x_token_test = heb_tokenizer.texts_to_sequences(model_data.x_token_test)

# pad
x_token_train, x_token_test = pad(x_token_train, x_token_test)







# ## Import required modules from Keras

# In[6]:


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
