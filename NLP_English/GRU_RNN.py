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
    EMBEDDING = 'Data/glove.840B.300d.txt'

    # Save training and testing Data
    TRAIN_DATA = 'Data/train.csv'
    TEST_DATA = 'Data/test.csv'
    LABELS_DATA = 'Data/test_labels.csv'
    SAMPLE_SUB = 'Data/sample_submission.csv'

    embed_size = 300  # Size of word vector, given by our pre-trained vectors
    max_features = 150000  # Number of unique words to use (i.e. num rows in embedding matrix)
    max_length = 100  # Max number of words in a comment to use

    # Load Data into pandas
    train = pd.read_csv(TRAIN_DATA)
    test = pd.read_csv(TEST_DATA)
    labels = pd.read_csv(LABELS_DATA)
    submission = pd.read_csv(SAMPLE_SUB)

    # Replace missing values in training and test set
    list_train = train["comment_text"].fillna("_na_").values
    classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y = train[classes].values
    list_test = test["comment_text"].fillna("_na_").values

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
    # nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((max_features, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # Bidirectional GRU with 2 layers, dropout, and max-pooling
    inp = Input(shape=(max_length,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    x = Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform", activation="relu")(x)
    # avg_pool = GlobalAveragePooling1D()(x)
    # max_pool = GlobalMaxPooling1D()(x)
    # x = concatenate([avg_pool, max_pool])
    x = Conv1D(64, kernel_size=6, padding="valid", kernel_initializer="he_uniform", activation="relu")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.1)(x)
    output = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')
    best_model = 'Models/model44.h5'
    checkpoint = ModelCheckpoint(best_model, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

    # Fit the model
    history = model.fit(X_t, y, batch_size=16, epochs=10, callbacks=[es, checkpoint], validation_split=0.1)

    plt.figure(figsize=(15, 8))
    plt.title("Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(history.history['val_loss'])
    plt.savefig("Images/GRU_2_ENG.jpg")

    model = load_model(best_model)
    print('**Predicting on test set**')
    prediction = model.predict(X_te, batch_size=16, verbose=1)

    y = labels.toxic
    fpr, tpr, _ = metrics.roc_curve(y, prediction[:, 1])

    # create ROC curve
    plt.plot(fpr, tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("Images/GRU_2_ROC")

    submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = prediction
    submission.to_csv('Predictions/submission18.csv', index=False)
