##2019 4 3 VAE GAN的更新版本，主要用于bar的旋律生成，以及后期的与gan的融合，还有就是加入主尾音
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, RepeatVector,Bidirectional
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import SGD, RMSprop, Adam
from keras import objectives
#from data_new import init_data

class vrae_muse():
    def __init__(self, input_dim,
                    timesteps,
                    batch_size,
                    intermediate_dim,
                    latent_dim,
                    epsilon_std=1.):
        self.input_dim = input_dim
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.epsilon_std = epsilon_std

        self.z_mean = None
        self.z_log_sigma = None
        self.original_dim = input_dim*timesteps

        self.generator = self.d_setup()
        self.encoder = self.e_setup()
        #self.discrimnator = self.dis_setup()

        self.vae = self.vae()

    def dis_setup(self):
        x = Input(shape=(self.timesteps, self.input_dim))
        dis = Model()
        return dis

    def e_setup(self):
        x = Input(shape=(self.timesteps, self.input_dim,))

        # LSTM encoding
        h = (LSTM(self.intermediate_dim))(x)
        #h = LSTM(self.intermediate_dim, dropout_W=0.2, dropout_U=0.2)(h)

        # VAE Z layer
        self.z_mean = Dense(self.latent_dim, name='mean')(h)
        self.z_log_sigma = Dense(self.latent_dim, name='sigma')(h)

        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = K.random_normal(shape=(self.batch_size, self.latent_dim),
                                      mean=0., stddev=self.epsilon_std )
            return z_mean + z_log_sigma * epsilon

        # note that "output_shape" isn't necessary with the TensorFlow backend
        # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
        z = Lambda(sampling, output_shape=(self.latent_dim,))([self.z_mean, self.z_log_sigma])

        encoder = Model(x, [z, self.z_mean, self.z_log_sigma])
        return encoder

    def d_setup(self):
        decoder_input = Input(shape=(self.latent_dim,))

        import numpy as np
        # decoded LSTM layer
        decoder_h = LSTM(self.intermediate_dim, return_sequences=True)
        decoder_mean = LSTM(self.input_dim, return_sequences=True)
        print("z:", np.shape(decoder_input))
        h_decoded = RepeatVector(self.timesteps)(decoder_input)
        print("h_d1", np.shape(h_decoded))
        h_decoded = decoder_h(h_decoded)
        print("h_d2", np.shape(h_decoded))
        # decoded layer
        x_decoded_mean = decoder_mean(h_decoded)

        def frisky(args):
            song = []

            for j in range(self.batch_size):
                s = []
                for i in range(self.timesteps):
                    #print("a", np.shape(args[:,i]))
                    t = K.sigmoid(args[j][i])
                    s.append(t)
                    #print("b", np.shape(t))

                song.append(s)
            song = K.reshape(song, shape=(self.batch_size,self.timesteps,self.input_dim))
            # song -= 0.5
            # song = K.sign(song)
            #song = K.permute_dimensions(song, (1, 0, 2))
            return song

        x_decoded_mean = Lambda(frisky)(x_decoded_mean)

        print("zzz", np.shape(h_decoded))
        print("zzz", np.shape(x_decoded_mean))
        # end-to-end autoencoder
        # vae = Model(x, x_decoded_mean)
        generator = Model(decoder_input, x_decoded_mean)
        return generator

    def vae(self):
        input_x = Input(shape=(self.timesteps, self.input_dim,))
        x = input_x
        z, z_mean, z_log_sigma = self.encoder(x)
        y = self.generator(z)

        def vae_loss(input_x, y):
            # xent_loss = 0
            # for i in range(self.timesteps):
            #     xent_loss += objectives.binary_crossentropy(input_x[:, i], y[:, i])
            # xent_loss = xent_loss /self.timesteps

            xent_loss = objectives.binary_crossentropy(input_x, y)
            #xent_loss = objectives.mse(input_x, y)
            #z_log_sigma = self.z_log_sigma
            #z_mean = self.z_mean
            kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
            loss = xent_loss + kl_loss
            return loss

        vae = Model(input_x, y)
        vae.compile(optimizer='rmsprop', loss=vae_loss)
        return vae

    def set_g(self, g):
        self.generator = g

    def set_e(self, e):
        self.encoder = e

    def set_v(self, g, e):
        input_x = Input(shape=(self.timesteps, self.input_dim,))
        x = input_x
        z, z_mean, z_log_sigma = e(x)
        y = g(z)

        def vae_loss(input_x, y):
            xent_loss = objectives.binary_crossentropy(input_x, y)
            #xent_loss = objectives.mse(input_x, y)
            # z_log_sigma = self.z_log_sigma
            # z_mean = self.z_mean
            kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
            loss = xent_loss + kl_loss
            return loss

        vae = Model(input_x, y)
        vae.compile(optimizer='rmsprop', loss=vae_loss)
        self.encoder = e
        self.generator = g
        self.vae = vae
        return vae

