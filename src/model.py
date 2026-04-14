from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_model():
    model = Sequential()

    model.add(Conv2D(24, (5, 5), strides=(2, 2),
              activation='relu', input_shape=(66, 200, 3)))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(Flatten())

    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))

    model.add(Dense(1))

    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss='mse', optimizer=optimizer)

    return model


def train(x_train, x_test, y_train, y_test, epochs, batch_size=1):
    model = build_model()
    h = model.fit(x_train, y_train, validation_data=(
        x_test, y_test), epochs=epochs, batch_size=batch_size)

    save_model(model, "../simulator/model.h5")
    return h
