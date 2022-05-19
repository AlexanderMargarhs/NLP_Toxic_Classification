import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.models import Model, load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, LSTM, Embedding, SpatialDropout1D, Dropout
from keras.layers import Bidirectional, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate


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

    # Bidirectional LSTM with 2 convolutional layers, max-pooling, and 1 FC layer
    inp = Input(shape=(max_length,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    x = Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform", activation="relu")(x)
    x = Conv1D(64, kernel_size=6, padding="valid", kernel_initializer="he_uniform", activation="relu")(x)

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)

    x = concatenate([avg_pool, max_pool])
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
    best_model = 'Models_GR/model42.h5'
    checkpoint = ModelCheckpoint(best_model, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

    # Fit the model
    history = model.fit(X_t, y, batch_size=16, epochs=10, callbacks=[es, checkpoint], validation_split=0.1)

    plt.figure(figsize=(15, 8))
    plt.title("Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(history.history['val_loss'])
    plt.savefig("Images/LSTM_2_GR")

    # Load best model
    model = load_model(best_model)
    print('**Predicting on test set**')
    prediction = model.predict(X_te, batch_size=16, verbose=1)
    submission[["subtask_a"]] = prediction
    submission.to_csv('Predictions_GR/submission16.csv', index=False)
