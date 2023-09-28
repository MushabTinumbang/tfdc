# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================
import csv

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import numpy as np


def solution_B4():
    bbc = pd.read_csv('https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or you can not pass this test
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8


    # YOUR CODE HERE
    # Using "shuffle=False"
    training_sentences, validation_sentences , training_labels, validation_labels = train_test_split(bbc['text'], bbc['category'],
                                                            train_size=training_portion, shuffle = False)


    # Fit your tokenizer with training data
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    sequences = tokenizer.texts_to_sequences(training_sentences)
    train_pad = pad_sequences(sequences, padding=padding_type, maxlen=max_length, truncating=trunc_type)

    test_seq = tokenizer.texts_to_sequences(validation_sentences)
    test_pad = pad_sequences(test_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)


    # You can also use Tokenizer to encode your label.

    all_labels = np.array(bbc['category'].values)
    label_token = Tokenizer()
    label_token.fit_on_texts(all_labels)

    train_labels = np.array(label_token.texts_to_sequences(training_labels))
    test_labels = np.array(label_token.texts_to_sequences(validation_labels))

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('val_accuracy') > 0.93 and logs.get('accuracy') > 0.95:
                self.model.stop_training = True

    callbacks = myCallback()


    model = tf.keras.Sequential([
        # YOUR CODE HERE.
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(30, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(train_pad, train_labels, epochs=500, validation_data=(test_pad, test_labels), callbacks=[callbacks])
    # Make sure you are using "sparse_categorical_crossentropy" as a loss fuction

    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B4()
    model.save("model_B4.h5")
