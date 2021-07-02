from __future__ import print_function
import keras
import numpy as np 
import tensorflow as tf

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D, Reshape
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras import backend as K


SEQ_LENGTH = 32


def sq_euclidean_distance(y_true, y_pred):
        squared_difference = tf.square(y_true - y_pred)
        sq_distance = tf.reduce_mean(squared_difference, axis=-1)  
        return sq_distance


def train():

    n = 100000
    x_train = np.zeros((n, SEQ_LENGTH))
    for i in range(n):
        x_train[i,:] = np.random.permutation(250)[0:SEQ_LENGTH]

    x_train = x_train.reshape(n, SEQ_LENGTH, 1)
    y_train = np.sort(x_train, axis=1).reshape(n, SEQ_LENGTH)

    n = 20000
    x_test = np.zeros((n,SEQ_LENGTH))
    for i in range(n):
        x_test[i,:] = np.random.permutation(250)[0:SEQ_LENGTH]

    x_test = x_test.reshape(n, SEQ_LENGTH, 1)
    y_test = np.sort(x_test, axis=1).reshape(n, SEQ_LENGTH)

    input_shape = (SEQ_LENGTH, 1)

    model = Sequential()

    model.add(Conv1D(32, kernel_size=(2),
                    activation='relu',
                    input_shape=input_shape,
                    padding='same'))
    model.add(Conv1D(32, (2), activation='relu', padding='same'))

    model.add(Conv1D(32, (4), activation='relu', padding='same'))
    model.add(Conv1D(32, (4), activation='relu', padding='same'))

    model.add(Conv1D(64, (8), activation='relu', padding='same'))
    model.add(Conv1D(64, (8), activation='relu', padding='same'))

    model.add(Conv1D(64, (16), activation='relu', padding='same'))
    model.add(Conv1D(64, (16), activation='relu', padding='same'))
    
    model.add(Conv1D(128, (32), activation='relu', padding='same'))
    model.add(Conv1D(128, (32), activation='relu', padding='same'))

    model.add(Flatten())
    model.add(Dense(SEQ_LENGTH))

    model.compile(loss=sq_euclidean_distance,
                optimizer=keras.optimizers.Adam(learning_rate=0.001))

    epochs = 400
    batch_size = 128

    history = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))

    best_loss = min(history.history["val_loss"])
    model.save(f"merge_sort_model_v14_{best_loss:.3f}.h5")

    test_list = np.random.permutation(150)[0:SEQ_LENGTH]
    pred = model.predict(test_list.reshape(1,SEQ_LENGTH,1))
    print(test_list)
    print(pred)

    print(np.sort(test_list).tolist())
    print([np.asarray(test_list).reshape(SEQ_LENGTH,)[np.abs(np.asarray(test_list).reshape(SEQ_LENGTH,) - i).argmin()] for i in list(pred[0])])

def test():

    model = keras.models.load_model('merge_sort_model_v14_3.398.h5', custom_objects={'sq_euclidean_distance': sq_euclidean_distance})
    test_list = np.random.permutation(150)[0:SEQ_LENGTH]
    pred = model.predict(test_list.reshape(1,SEQ_LENGTH,1))

    print("############################################################")
    print(test_list.tolist(), ": Original Array")
    print(np.sort(test_list).tolist(), ": Target Array")

    cleaned_pred = [np.asarray(test_list).reshape(SEQ_LENGTH,)
                   [np.abs(np.asarray(test_list).reshape(SEQ_LENGTH,) - i).argmin()] 
                   for i in list(pred[0])]
    print(cleaned_pred, ": NN Output")

    print("\n\n")


if __name__ == "__main__":
    # train()
    for i in range(10):
        test()