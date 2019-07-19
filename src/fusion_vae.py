##2019 4 3 VAE GAN的更新版本，主要用于bar的旋律生成，以及后期的与gan的融合，还有就是加入主尾音
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, RepeatVector,Bidirectional
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import SGD, RMSprop, Adam
from keras import objectives
#from data_new import init_data
import numpy as np


def input_trans(input):  ##input shape = (num, track, dim)
    ## we need concat this dim
    ##concat
    input_trans = np.reshape(input, newshape=[input.shape[0], input.shape[1] * input.shape[2]])

    return input_trans


def output_trans(output, num_track):
    output_trans = np.reshape(output, newshape=[output.shape[0], num_track, output.shape[1] // num_track])
    return output_trans

class vrae_fusion():
    '''
    策略有以下几种：
    1.直接对5个维度的隐向量cat 不收敛
    2.直接对5个维度的隐向量LSTM  可以（目前最好），但是不太理想。
    3.直接对5个维度的隐向量cat+add （试一下）
    4.直接对5个维度的隐向量CCA，多模态融合.
    5.非对称多模态融合-》生成网络
    6.共享隐空间 试一下
    '''
    def __init__(self, input_dim,
                    batch_size,
                    track_num,
                    latent_dim,
                    epsilon_std=1.,
                    flag = 1):
        ##flag == 1, 2, 3, 4, 5, 6
        '''
        :param input_dim:
        :param batch_size:
        :param track_num:
        :param latent_dim:
        :param epsilon_std:
        :param flag:  1代表cat，2代表add 3代表cat+add同时
        '''

        self.input_dim = input_dim
        self.batch_size = batch_size
        self.track_num = track_num
        self.latent_dim = latent_dim
        self.epsilon_std = epsilon_std

        self.input_shape = track_num*input_dim
        print("INPUT SHAPE STAGE2", self.input_shape)
        self.z_mean = None
        self.z_log_sigma = None

        if(flag == 1):## cat的方式

            self.generator = self.cat_d_setup()
            self.encoder = self.cat_e_setup()
            self.vae = self.cat_vae()
        elif(flag == 2): ## add的方式 非对称

            self.generator = self.add_d_setup()
            self.encoder = self.add_e_setup()
            self.vae = self.add_vae()

    def add_d_setup(self):
        return None

    def add_e_setup(self):
        x = Input(shape=(self.input_shape,))

        # LSTM encoding
        h = Dense(128, activation='relu')(x)
        h = Dense(128, activation='relu')(h)
        h = Dense(64, activation='relu')(h)
        h = Dense(64, activation='relu')(h)
        h = Dense(32, activation='relu')(h)
        h = Dense(32, activation='relu')(h)
        # h = LSTM(self.intermediate_dim, dropout_W=0.2, dropout_U=0.2)(h)

        # VAE Z layer
        self.z_mean = Dense(self.latent_dim, name='mean')(h)
        self.z_log_sigma = Dense(self.latent_dim, name='sigma')(h)

        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = K.random_normal(shape=(self.batch_size, self.latent_dim),
                                      mean=0., stddev=self.epsilon_std)
            return z_mean + z_log_sigma * epsilon

        # note that "output_shape" isn't necessary with the TensorFlow backend
        # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
        z = Lambda(sampling, output_shape=(self.latent_dim,))([self.z_mean, self.z_log_sigma])

        encoder = Model(x, [z, self.z_mean, self.z_log_sigma])
        return encoder
    def add_vae(self):
        return None

    def cat_e_setup(self):
        x = Input(shape=(self.input_shape, ))

        # LSTM encoding
        h = Dense(2048, activation='relu')(x)
        h = Dense(2048, activation='relu')(h)
        h = Dense(1024, activation='relu')(h)
        h = Dense(1024, activation='relu')(h)
        h = Dense(512, activation='relu')(h)
        h = Dense(256, activation='relu')(h)
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

    def cat_d_setup(self):
        decoder_input = Input(shape=(self.latent_dim,))

        h = Dense(256, activation='relu')(decoder_input)
        h = Dense(512, activation='relu')(h)
        h = Dense(1024, activation='relu')(h)
        h = Dense(1024, activation='relu')(h)
        h = Dense(2048, activation='relu')(h)
        h = Dense(2048, activation='relu')(h)

        # decoded layer
        x_decoded_mean = Dense(self.track_num*self.input_dim, activation='relu')(h)

        #print("zzz", np.shape(x_decoded_mean))
        # end-to-end autoencoder
        # vae = Model(x, x_decoded_mean)
        generator = Model(decoder_input, x_decoded_mean)
        return generator

    def cat_vae(self):
        input_x = Input(shape=(self.input_shape,))
        x = input_x
        z, z_mean, z_log_sigma = self.encoder(x)
        y = self.generator(z)

        def vae_loss(input_x, y):
            # xent_loss = objectives.binary_crossentropy(input_x, y)
            xent_loss = objectives.mse(input_x, y)
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
        input_x = Input(shape=(self.input_shape,))
        x = input_x
        z, z_mean, z_log_sigma = e(x)
        y = g(z)

        def vae_loss(input_x, y):
            xent_loss = objectives.mse(input_x, y)
            # z_log_sigma = self.z_log_sigma
            # z_mean = self.z_mean
            kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
            loss = xent_loss + kl_loss
            return loss

        vae = Model(input_x, y)
        vae.compile(optimizer='rmsprop', loss=vae_loss)
        self.encoder = e
        self.generator = g
        self.cat_vae = vae
        return vae

