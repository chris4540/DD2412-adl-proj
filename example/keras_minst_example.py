"""
Example to use new model subclass API
"""
import time
import numpy as np
from matplotlib import pyplot as plt
import tensorflow.keras.utils as utils
import tensorflow.keras.callbacks as cb
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.datasets import mnist


class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)

class FeedFwdNetwork(Model):
    """
    Simple network for testing keras/ tensorflow
    """
    def __init__(self):
        super(FeedFwdNetwork, self).__init__()
        self.fc1 = Dense(500, input_dim=784)
        self.relu1 = Activation('relu')
        self.dropout1 = Dropout(0.4)

        self.fc2 = Dense(300)
        self.relu2 = Activation('relu')
        self.dropout2 = Dropout(0.4)

        self.fc3 = Dense(10)
        self.softmax = Activation('softmax')

    def call(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        out = self.fc3(out)
        out = self.softmax(out)
        return out



def load_data():
    print('Loading data...')
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255

    y_train = utils.to_categorical(y_train, 10)
    y_test = utils.to_categorical(y_test, 10)

    X_train = np.reshape(X_train, (60000, 784))
    X_test = np.reshape(X_test, (10000, 784))

    print('Data loaded.')
    return [X_train, X_test, y_train, y_test]


def init_model():
    start_time = time.time()
    print('Compiling Model ... ')
    # model = Sequential()
    # model.add(Dense(500, input_dim=784))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.4))
    # model.add(Dense(300))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.4))
    # model.add(Dense(10))
    # model.add(Activation('softmax'))
    model = FeedFwdNetwork()

    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    print('Model compield in {0} seconds'.format(time.time() - start_time))
    return model


def run_network(data=None, model=None, epochs=20, batch=256):
    try:
        start_time = time.time()
        if data is None:
            X_train, X_test, y_train, y_test = load_data()
        else:
            X_train, X_test, y_train, y_test = data

        if model is None:
            model = init_model()

        history = LossHistory()

        print('Training model...')
        model.fit(X_train, y_train, nb_epoch=epochs, batch_size=batch,
                  callbacks=[history],
                  validation_data=(X_test, y_test), verbose=2)

        print("Training duration : {0}".format(time.time() - start_time))
        score = model.evaluate(X_test, y_test, batch_size=16)

        print("Network's test score [loss, accuracy]: {0}".format(score))
        return model, history.losses
    except KeyboardInterrupt:
        print(' KeyboardInterrupt')
        return model, history.losses


def plot_losses(losses):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(losses)
    ax.set_title('Loss per batch')
    fig.show()

if __name__ == "__main__":
    # model, losses = run_network()
    model = FeedFwdNetwork()
    # model.build(input_shape=(10, 784))
    model.predict()
    model.summary()