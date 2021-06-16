# Auxiliary Classifier Generative Adversarial Network

from keras.layers import Input, Dense, Reshape, Dropout, Convolution2D, Convolution2DTranspose, UpSampling2D, multiply
from keras.layers import BatchNormalization, Activation, Flatten, MaxPooling1D, MaxPooling2D, ZeroPadding2D, Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import np_utils
import keras.backend as K
import numpy as np
import tensorflow as tf


class GAN():
    def __init__(self, datashape, num_classes):
        self.gan = None

        ### Generator
        self.generator = None
        self.generator_optimizer = Adam(0.0002, 0.9)

        ### Discriminator
        self.discriminator = None
        self.discriminator_optimizer = SGD(lr=0.012)

        self.num_samples = datashape[0]
        self.features = datashape[1]
        self.seq_length = datashape[2]

        self.input_shape = (self.features, self.seq_length, 1)
        self.num_classes = 10000
        self.latent_dim = 10000 + 100

        ### Build network
        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()
        self.gan = self.build_gan()

    def build_gan(self):
        self.discriminator.trainable = False
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.discriminator_optimizer, metrics=['accuracy', self.recall, self.precision, self.fmeasure])
        # z = Input(shape=(self.latent_dim,))

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))

        generated_seq = self.generator([noise, label])
        valid, target_label = self.discriminator(generated_seq)

        model = Model([noise, label], [valid, target_label])
        model.compile(loss='binary_crossentropy', optimizer=self.generator_optimizer)

        return model

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 1 * 20, activation="relu", input_dim=self.latent_dim))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Reshape((1, 20, 128)))
        model.add(UpSampling2D(size=(1, 5)))
        model.add(Convolution2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D(size=(1, 2)))
        model.add(Convolution2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Convolution2D(1, kernel_size=3, padding='same'))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Convolution2D(16, kernel_size=3, strides=2, input_shape=self.input_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Convolution2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Convolution2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Convolution2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.summary()

        img = Input(shape=self.input_shape)

        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(1, activation="softmax")(features)

        return Model(img, [validity, label])

    def generate(self, sampled_labels):
        noise = np.random.normal(0, 1, (128, self.latent_dim))
        predictions = self.generator.predict([noise, sampled_labels])
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
