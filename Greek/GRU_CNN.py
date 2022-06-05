import numpy as np
import pandas as pd
from sklearn import metrics
from matplotlib import pyplot as plt
from keras.models import Model, load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, Embedding, Dropout, SpatialDropout1D, concatenate
from keras.layers import GRU, Conv1D, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D


# Read word vectors into a dictionary
def get_coefficiency(Word, *arr):
	return Word, np.asarray(arr, dtype='float32')


if __name__ == "__main__":
	# Load pre-trained word vectors
	EMBEDDING = 'Data_GR/glove.840B.300d.txt'

	# Save training and testing Data
	TRAIN_DATA = 'Data_GR/train.csv'
	TEST_DATA = 'Data_GR/test.csv'
	SAMPLE_SUB = 'Data_GR/labels.csv'

	embed_size = 300  # Size of word vector, given by our pre-trained vectors
	max_features = 150000  # Number of unique words to use (i.e. num rows in embedding matrix)
	max_length = 100  # Max number of words in a comment to use

	# Load Data into pandas
	train = pd.read_csv(TRAIN_DATA)
	test = pd.read_csv(TEST_DATA)
	submission = pd.read_csv(SAMPLE_SUB)

	# Replace missing values in training and test set
	list_train = train["tweet"].fillna("_na_").values
	classes = ["subtask_a"]
	y = train[classes].values
	list_test = test["tweet"].fillna("_na_").values

	# Use Keras preprocessing tools
	tok = Tokenizer(num_words=max_features)
	tok.fit_on_texts(list(list_train))
	tokenized_train = tok.texts_to_sequences(list_train)
	tokenized_test = tok.texts_to_sequences(list_test)

	# Pad vectors with 0s for sentences shorter than max length
	X_t = pad_sequences(tokenized_train, maxlen=max_length)
	X_te = pad_sequences(tokenized_test, maxlen=max_length)

	embeddings_index = dict(get_coefficiency(*o.strip().split(" ")) for o in open(EMBEDDING, encoding="utf8"))

	# Create the embedding matrix
	word_index = tok.word_index

	embedding_matrix = np.zeros((max_features, embed_size))
	for word, i in word_index.items():
		if i >= max_features:
			continue
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector

	# Bidirectional GRU-CNN with max-pooling and 2 FC layers
	inp = Input(shape=(max_length,))
	x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
	x = SpatialDropout1D(0.2)(x)
	x = Bidirectional(GRU(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
	x = Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform", activation="relu")(x)

	avg_pool = GlobalAveragePooling1D()(x)
	max_pool = GlobalMaxPooling1D()(x)

	x = concatenate([avg_pool, max_pool])
	x = Dense(128, activation="relu")(x)
	x = Dropout(0.1)(x)
	x = Dense(64, activation="relu")(x)
	x = Dropout(0.1)(x)

	output = Dense(1, activation="sigmoid")(x)

	model = Model(inputs=inp, outputs=output)
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	es = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
	best_model = 'Models_GR/model43.h5'
	checkpoint = ModelCheckpoint(best_model, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

	# Fit the model
	history = model.fit(X_t, y, batch_size=16, epochs=10, callbacks=[es, checkpoint], validation_split=0.1)

	plt.figure(figsize=(15, 8))
	plt.title("Training loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.plot(history.history['val_loss'])
	plt.savefig("Images/GRU_1_GR")

	model = load_model(best_model)
	print('**Predicting on test set**')
	prediction_te = model.predict(X_te, batch_size=16, verbose=1)
	prediction_te = np.where(prediction_te > 0.5, 1, 0)

	prediction_tr = model.predict(X_t, batch_size=16, verbose=1)
	prediction_tr = np.where(prediction_tr > 0.5, 1, 0)

	y_te = submission.subtask_a
	y_tr = train.subtask_a

	pr_te = metrics.precision_score(y_te, prediction_te, average='micro')
	f1_te = metrics.f1_score(y_te, prediction_te, average='micro')
	re_te = metrics.recall_score(y_te, prediction_te, average='micro')

	pr_tr = metrics.precision_score(y_tr, prediction_tr, average='micro')
	f1_tr = metrics.f1_score(y_tr, prediction_tr, average='micro')
	re_tr = metrics.recall_score(y_tr, prediction_tr, average='micro')

	# score
	score_train = metrics.accuracy_score(y_tr, prediction_tr)
	score_test = metrics.accuracy_score(y_te, prediction_te)

	# summarize
	print('Accuracy: train=%.3f' % (score_train * 100))
	print('Precision: train=%.3f' % (pr_tr * 100))
	print('F1-Score: train=%.3f' % (f1_tr * 100))
	print('Recall: train=%.3f' % (re_tr * 100))

	print('Accuracy: test=%.3f' % (score_test * 100))
	print('Precision: test=%.3f' % (pr_te * 100))
	print('F1-Score: test=%.3f' % (f1_te * 100))
	print('Recall: test=%.3f' % (re_te * 100))

	# Compute micro-average ROC curve and ROC area
	fpr, tpr, _ = metrics.roc_curve(y_te.ravel(), prediction_te.ravel())
	roc_auc = metrics.auc(fpr, tpr)

	# create ROC curve
	plt.figure()
	lw = 2
	plt.plot(
		fpr,
		tpr,
		color="darkorange",
		lw=lw,
		label="ROC curve (area = %0.2f)" % roc_auc,
	)

	plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.title("Receiver operating characteristic example")
	plt.legend(loc="lower right")
	plt.savefig("Images/GRU_1_ROC")

	submission[["submission"]] = prediction_te
	submission.to_csv('Predictions_GR/submission17.csv', index=False)
