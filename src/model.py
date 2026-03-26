from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam

def build_model():
    model = Sequential()

    model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(120,120,1)))
    model.add(MaxPooling2D())

    model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
