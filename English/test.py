import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

if __name__ == "__main__":
	EMBEDDING_FILE = 'Data/glove.6B.50d.txt'  # Computed a GloVe embedding from corpus
	TRAIN_DATA_FILE = 'Data/train.csv'  # Training Data
	TEST_DATA_FILE = 'Data/test.csv'  # Testing Data

	models = ['Models/model41.h5', 'Models/model42.h5', 'Models/model43.h5', 'Models/model44.h5']

	for model in models:

		Model = load_model(model)

		embed_size = 50  # Size of word vector
		max_features = 20000  # Number of unique words to use (i.e. num rows in embedding vector)
		max_length = 100  # Max number of words in a comment to use

		# Load Data into pandas
		train = pd.read_csv(TRAIN_DATA_FILE)
		test = pd.read_csv(TEST_DATA_FILE)

		# Replace missing values in training and test set
		list_sentences_train = train["comment_text"].fillna("_na_").values
		list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
		y = train[list_classes].values
		list_sentences_test = test["comment_text"].fillna("_na_").values

		# Use Keras preprocessing tools
		tokenizer = Tokenizer(num_words=max_features)
		tokenizer.fit_on_texts(list(list_sentences_train))
		list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
		list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
		X_t = pad_sequences(list_tokenized_train, maxlen=max_length)
		X_te = pad_sequences(list_tokenized_test, maxlen=max_length)

		y_train = Model.predict([X_t], batch_size=16, verbose=1)

		toxic_i = np.argsort(y_train[:, -1])

		print("Using Model:", model)
		for i in toxic_i:
			print(list_sentences_train[i])
