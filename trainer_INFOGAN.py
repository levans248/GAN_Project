

import numpy as np
from utils import get_batch, sample_image
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from keras.utils import to_categorical
import time


class Trainer():
    def __init__(self, model, PATH):
        self.model = model
        self.disc_loss = []

        self.disc_loss_r = []
        self.disc_loss_f = []

        self.crit_loss_r = []
        self.crit_loss_f = []

        self.crit_loss = []
        self.gen_loss = []
        self.path = PATH

    def train_gan(self, epochs, batch_size, sample_interval, train_data):

        # Create labels for real and fake data
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        start_time = time.time()
        for epoch in range(epochs):

             # Select a random half batch of images
            idx = np.random.randint(0, train_data.shape[0], batch_size)
            imgs = train_data[idx]

            # Sample noise and categorical labels
            sampled_noise, sampled_labels = self.sample_generator_input(batch_size, train_data)
            gen_input = np.concatenate((sampled_noise, sampled_labels), axis=1)

            # Generate a half batch of new images
            gen_imgs = self.model.generator.predict(gen_input)

            # Train on real and generated data
            d_loss_real = self.model.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.model.discriminator.train_on_batch(gen_imgs, fake)

            # Avg. loss
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator and Q-network
            # ---------------------

            g_loss = self.model.combined.train_on_batch(gen_input, [valid, sampled_labels])

            # perplexity = e^loss
            train_perplexity = np.exp(g_loss[0])

            # Plot the progress
            print ("%d [D loss: %.2f, acc.: %.2f%%] [Q loss: %.2f] [G loss: %.2f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[1], g_loss[2]))

            if epoch % sample_interval == 0:
                sample_time = time.time() - start_time
                print("""%d [DiscLoss/Acc Real: (%10f, %10f)]
                       [Disc recall/precision/fmeasure Real: (%10f, %10f, %10f)]
                       [DiscLoss/Acc Fake: (%10f, %10f)]
                       [Disc recall/precision/fmeasure Fake: (%10f, %10f, %10f)]  
                       [DiscAcc %10f][GenLoss = %10f, GenPerplexity = %10f]
                       [Time taken during sample interval (seconds): %10f]"""
                      % (epoch, d_loss_real[0], d_loss_real[1],
                         d_loss_real[2], d_loss_real[3], d_loss_real[4],
                         d_loss_fake[0], d_loss_fake[1],
                         d_loss_fake[2], d_loss_fake[3], d_loss_fake[4],
                         0.5 * (d_loss_real[1] + d_loss_fake[1]),
                         g_loss[0], train_perplexity, sample_time))

                self.disc_loss_r.append(d_loss_real)
                self.disc_loss_f.append(d_loss_fake)

                self.gen_loss.append(g_loss[0])
                sample_image(self.model, epoch, gen_imgs, self.path)
                start_time = time.time()
            if (epoch % 1000 == 0):
                self.save_models(self.path, epoch, self.model.generator, self.model.discriminator)

        self.savedata(self.path, train_data)
        self.showLoss(self.path, save=True)

    def save_models(self, path, epoch, generator, discriminator=None, critic=None):
        generator.save('{}/models/generator_{}.h5'.format(path, epoch))

        if discriminator:
            discriminator.save('{}/models/discriminator_{}.h5'.format(path, epoch))
        if critic:
            critic.save('{}/models/critic_{}.h5'.format(path, epoch))

    def savedata(self, path, train_set):
        print("saving")
        fn1 = path + "/gen_loss.npy"

        fn2 = path + "/disc_loss_r.npy"
        fn3 = path + "/disc_loss_f.npy"

        fn4 = path + "/crit_loss_r.npy"
        fn5 = path + "/crit_loss_f.npy"

        np.save(fn1, self.gen_loss)

        np.save(fn2, self.disc_loss_r)
        np.save(fn3, self.disc_loss_f)

        np.save(fn4, self.crit_loss_r)
        np.save(fn5, self.crit_loss_f)

    def showLoss(self, path, save=True):
        onRealLoss = np.array(self.disc_loss_r)[:, 0]
        onRealAcc = np.array(self.disc_loss_r)[:, 1]

        onFakeLoss = np.array(self.disc_loss_f)[:, 0]
        onFakeAcc = np.array(self.disc_loss_f)[:, 1]

        g_Loss = np.array(self.gen_loss)
        plt.figure(figsize=(8, 5.5), dpi=100)

        plt.title("Discriminator and Generated Losses")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

        plt.plot([-10, 1010], [np.log(2), np.log(2)], '-', label="Expected value (log 2)", c='k', lw=3)

        plt.plot(savgol_filter(onRealLoss, 11, 1), label="Discriminator on Real", c='g')
        plt.plot(savgol_filter(onFakeLoss, 11, 1), label="Discriminator on Fake", c='r')
        plt.plot(savgol_filter(g_Loss, 15, 1), label="Generator", c='b')
        plt.grid()
        plt.legend()

        if save == True:
            plt.savefig(path + "/plots/Losses.png", edgecolor='k', dpi=100)

        plt.figure(figsize=(8, 6), dpi=100)
        plt.title("Accuracy of Discriminator in Correctly Identifying Real and Fake samples")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")

        if onFakeAcc.size % 2 == 0:
            windowLength = onFakeAcc.size - 1
        else:
            windowLength = onFakeAcc.size

        plt.plot(savgol_filter(onFakeAcc, windowLength, 3), label="Fake", c='r')
        plt.plot(savgol_filter(onRealAcc, windowLength, 3), label="Real", c='g')
        plt.plot([-10, 1010], [0.5, 0.5], '-', label="Expected value (0.5)", c='k', lw=3)
        plt.grid()
        plt.legend()
        if save == True:
            plt.savefig(path + "/plots/Accuracies.png", edgecolor='k', dpi=100)

    def sample_generator_input(self, batch_size, data):
        # Generator inputs
        sampled_noise = np.random.normal(0, 1, (batch_size, 100))
        sampled_labels = np.random.randint(0, self.model.num_classes, batch_size).reshape(-1, 1)
        sampled_labels = to_categorical(sampled_labels, num_classes=self.model.num_classes)

        return sampled_noise, sampled_labels
