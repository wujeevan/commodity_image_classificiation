from keras.models import Model
from keras.layers import Input, Flatten, Dropout
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16


class Models(object):
    def __init__(self):
        pass

    @staticmethod
    def mnist_digit_fc(input_shape=(28, 28, 1), classes=10):
        img_input = Input(input_shape)

        x = Flatten(name='flatten')(img_input)
        x = Dense(64, activation='relu', name='dense1')(x)
        x = Dense(32, activation='relu', name='dense2')(x)
        x = Dense(classes, activation='softmax', name='prediction')(x)

        model = Model(img_input, x, name='mnist_digit_fc')
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    @staticmethod
    def mnist_digit(input_shape=(28, 28, 1), classes=10):
        img_input = Input(input_shape)

        x = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv1')(img_input)
        x = MaxPooling2D((2, 2), name='pool1')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2')(x)
        x = MaxPooling2D((2, 2), name='pool2')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv3')(x)

        x = Flatten()(x)
        x = Dense(128, activation='relu', name='fc1')(x)
        x = Dropout(0.2, name='drop1')(x)
        x = Dense(classes, activation='softmax', name='prediction')(x)

        model = Model(img_input, x, name='mnist_digit')
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    @staticmethod
    def mnist_fashion(input_shape=(28, 28, 1), classes=10):
        img_input = Input(input_shape)

        x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1')(img_input)
        x = MaxPooling2D((2, 2), name='pool1')(x)
        x = Conv2D(48, (3, 3), activation='relu', padding='same', name='conv2')(x)
        x = MaxPooling2D((2, 2), name='pool2')(x)
        x = Conv2D(64, (2, 2), activation='relu', padding='same', name='conv3')(x)

        x = Flatten()(x)
        x = Dense(128, activation='relu', name='fc1')(x)
        x = Dropout(0.2, name='drop1')(x)
        x = Dense(classes, activation='softmax', name='prediction')(x)

        model = Model(img_input, x, name='mnist_digit')
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    @staticmethod
    def cifar10(input_shape=(32, 32, 3), classes=10):
        img_input = Input(input_shape)

        x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1')(img_input)
        x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv2')(x)
        x = MaxPooling2D((2, 2), name='pool1')(x)
        x = Dropout(0.25, name='drop1')(x)

        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv3')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv4')(x)
        x = MaxPooling2D((2, 2), name='pool2')(x)
        x = Dropout(0.25, name='drop2')(x)

        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv5')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv6')(x)
        x = MaxPooling2D((2, 2), name='pool3')(x)
        x = Dropout(0.25, name='drop3')(x)

        x = Flatten()(x)
        x = Dense(256, activation='relu', name='fc1')(x)
        x = Dropout(0.25, name='drop4')(x)
        x = Dense(classes, activation='softmax', name='prediction')(x)

        model = Model(img_input, x, name='mnist_digit')
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    @staticmethod
    def daily(input_shape=(224, 224, 3), classes=40, model_path='../models/vgg16_weights_notop.h5'):
        base_model = VGG16(weights=model_path, include_top=False, input_shape=input_shape)

        x = base_model.output
        # x = Flatten()(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=x)

        for layer in base_model.layers:
            layer.trainable = False

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    @staticmethod
    def conv_vgg16(input_shape=(224, 224, 3), classes=40, model_path='../models/vgg16_weights_notop.h5'):
        base_model = VGG16(weights=model_path, include_top=False, input_shape=input_shape)
        x = base_model.output
        model = Model(input=base_model.input, outputs=x)
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model





















