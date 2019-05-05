from keras.preprocessing import text, sequence
import pickle


def load_tokinizer():
    with open('..\\pickles\\tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        return tokenizer


class HebrewTokenizer:

    def __init__(self, data, vocabulary_size=5000, char_level=False, init_from_file=False):
        if init_from_file:
            self.tokenizer = load_tokinizer()
            print("tokenizer loaded from file")
        else:
            tokenize = text.Tokenizer(num_words=vocabulary_size,
                                      char_level=char_level,
                                      filters='')
            tokenize.fit_on_texts(data)
            self.tokenizer = tokenize
            self.save_model()

    def texts_to_sequences(self, data):
        data = self.tokenizer.texts_to_sequences(data)
        return data

    def sequences_to_texts(self, data):
        texts = self.tokenizer.sequences_to_matrix(data)
        return texts

    def save_model(self):
        # saving
        with open('..\\pickles\\tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
