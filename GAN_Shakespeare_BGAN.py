# Boundary-Seeking Generative Adversarial Networks

from keras.layers import Input, Dense, Reshape, Dropout, Convolution2D, Conv2DTranspose, UpSampling2D
from keras.layers import BatchNormalization, Activation, Flatten, MaxPooling1D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import np_utils
import keras.backend as K
import numpy as np
import tensorflow as tf


class GAN():
    def __init__(self, datashape):
        self.gan = None

        ### Generator
        self.generator = None
        self.generator_optimizer = Adam(0.0002, 0.5)

        ### Discriminator
        self.discriminator = None
        self.discriminator_optimizer = SGD(lr=0.012)

        self.num_samples = datashape[0]
        self.features = datashape[1]
        self.seq_length = datashape[2]

        self.input_shape = (self.features, self.seq_length, 1)
        self.latent_dim = 100

        ### Build network
        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()
        self.gan = self.build_gan()

    def build_gan(self):
        self.discriminator.trainable = False
        z = Input(shape=(self.latent_dim,))
        generated_seq = self.generator(z)
        validity = self.discriminator(generated_seq)

        model = Model(z, validity)
        model.compile(loss=self.boundary_loss, optimizer=self.generator_optimizer)

        return model

    def build_discriminator(self):
        model = Sequential()
        model.add(Convolution2D(50, (1, 10), padding='valid', activation='relu', input_shape=self.input_shape))
        model.add(Convolution2D(50, (1, 5), padding='same', activation='relu'))

        model.add(Dense(int(50), activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, 2)))
        model.add(Dropout(0.2))

        model.add(Convolution2D(40, (1, 3), padding='valid', activation='relu'))
        model.add(Dense(40, activation='relu'))
        model.add(MaxPooling2D(pool_size=(1, 3)))
        model.add(Dropout(0.2))

        model.add(Convolution2D(20, (1, 3), padding='valid', activation='relu'))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense((400), activation='relu'))
        model.add(Dropout(0.4))

        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        model.compile(loss='binary_crossentropy', optimizer=self.discriminator_optimizer, metrics=['accuracy', self.recall, self.precision, self.fmeasure])

        return model

    def build_generator(self):
        model = Sequential()
        model.add(Dense(1024, input_dim=self.latent_dim, ))
        model.add(BatchNormalization())

        model.add(LeakyReLU(alpha=0.2))

        #model.add(Dense(128 * 3 * 20))
        model.add(Dense(128 * 1 * 20))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        #model.add(Reshape((3, 20, 128)))
        model.add(Reshape((1, 20, 128)))

        model.add(UpSampling2D(size=(1, 5)))
        model.add(Convolution2D(128, (1, 3), strides=(1, 1), padding='same'))
        model.add(LeakyReLU(alpha=0.2))

        model.add(UpSampling2D(size=(1, 2)))
        model.add(Convolution2D(1, (1, 5), strides=(1, 1), padding='same'))
        model.add(Activation('tanh'))

        model.summary()
        return model

    def boundary_loss(self, y_true, y_pred):
        """
        Boundary seeking loss.
        Reference: https://wiseodd.github.io/techblog/2017/03/07/boundary-seeking-gan/
        """
        return 0.5 * K.mean((K.log(y_pred) - K.log(1 - y_pred))**2)

    def generate(self, sampled_labels = False):
        noise = np.random.normal(0, 1, (1, self.latent_dim))
        predictions = self.generator.predict(noise)
        return predictions

    def recall(self, y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(self, y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def fbeta_score(self, y_true, y_pred, beta=1):
        """Computes the F score.
        The F score is the weighted harmonic mean of precision and recall.
        Here it is only computed as a batch-wise average, not globally.
        This is useful for multi-label classification, where input samples can be
        classified as sets of labels. By only using accuracy (precision) a model
        would achieve a perfect score by simply assigning every class to every
        input. In order to avoid this, a metric should penalize incorrect class
        assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
        computes this, as a weighted mean of the proportion of correct class
        assignments vs. the proportion of incorrect class assignments.
        With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
        correct classes becomes more important, and with beta > 1 the metric is
        instead weighted towards penalizing incorrect class assignments.
        """

        if beta < 0:
            raise ValueError('The lowest choosable beta is zero (only precision).')

        # If there are no true positives, fix the F score at 0 like sklearn.
        if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
            return 0

        p = self.precision(y_true, y_pred)
        r = self.recall(y_true, y_pred)
        bb = beta ** 2
        fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
        return fbeta_score

    def fmeasure(self, y_true, y_pred):
        """Computes the f-measure, the harmonic mean of precision and recall.
        Here it is only computed as a batch-wise average, not globally.
        """
        return self.fbeta_score(y_true, y_pred, beta=1)

