"""This script trains a model."""
import os
import logging
import argparse
from pprint import pformat
import numpy as np
import scipy.stats
import tensorflow as tf
from musegan.config import LOGLEVEL, LOG_FORMAT
from musegan.data import load_data, get_dataset, get_samples
from metrics import *
from musegan.model import Model
from musegan.utils import make_sure_path_exists, load_yaml
from musegan.utils import backup_src, update_not_none, setup_loggers
from write_midi import *
LOGGER = logging.getLogger("musegan.train")
from bar_vae import vrae_muse
from fusion_vae import vrae_fusion, input_trans, output_trans
from pypianoroll import Multitrack, Track
from resnet import BN_REFINE
import pypianoroll
import imageio
from loss_history import LossHistory






NUM_SAMPLE = 3200
TRAIN_STAGE1 = False
GEN_STAGE1 = False
BINARY_OUTPUT = False
TRAIN_STAGE2 = False
TEST_STAGE2 = True
BN_REFINE_TRAIN = False

#DATA_STRINT = "19_7_8_" # 6666的版本
#DATA_STRINT = "19_7_8_2_" ##binary_crossentropy
DATA_STRINT = "19_7_9_" # 9999的版本
DATA_STRING = "19_7_11_1_" ##最近成熟版
#DATA_STRING = "19_7_15_1_" ##最新测试版
def get_image_grid(images, shape, grid_width=0, grid_color=0,
                   frame=False):
    """
    Merge the input images and return a merged grid image.

    Arguments
    ---------
    images : np.array, ndim=3
        The image array. Shape is (num_image, height, width).
    shape : list or tuple of int
        Shape of the image grid. (height, width)
    grid_width : int
        Width of the grid lines. Default to 0.
    grid_color : int
        Color of the grid lines. Available values are 0 (black) to
        255 (white). Default to 0.
    frame : bool
        True to add frame. Default to False.

    Returns
    -------
    merged : np.array, ndim=3
        The merged grid image.
    """
    #print("get image grid", np.shape(images), shape)
    # exit(0)
    reshaped = images.reshape(shape[0], shape[1], images.shape[1],
                              images.shape[2])
    pad_width = ((0, 0), (0, 0), (grid_width, 0), (grid_width, 0))
    padded = np.pad(reshaped, pad_width, 'constant', constant_values=grid_color)
    transposed = padded.transpose(0, 2, 1, 3)
    merged = transposed.reshape(shape[0] * (images.shape[1] + grid_width),
                                shape[1] * (images.shape[2] + grid_width))
    if frame:
        return np.pad(merged, ((0, grid_width), (0, grid_width)), 'constant',
                      constant_values=grid_color)
    return merged[grid_width:, grid_width:]


def save_image(filepath, phrases, shape, inverted=True, grid_width=3,
               grid_color=0, frame=True):
    """
    Save a batch of phrases to a single image grid.

    Arguments
    ---------
    filepath : str
        Path to save the image grid.
    phrases : np.array, ndim=5
        The phrase array. Shape is (num_phrase, num_bar, num_time_step,
        num_pitch, num_track).
    shape : list or tuple of int
        Shape of the image grid. (height, width)
    inverted : bool
        True to invert the colors. Default to True.
    grid_width : int
        Width of the grid lines. Default to 3.
    grid_color : int
        Color of the grid lines. Available values are 0 (black) to
        255 (white). Default to 0.
    frame : bool
        True to add frame. Default to True.
    """
    if phrases.dtype == np.bool_:
        if inverted:
            phrases = np.logical_not(phrases)
        clipped = (phrases * 255).astype(np.uint8)
    else:
        if inverted:
            phrases = 1. - phrases
        clipped = (phrases * 255.).clip(0, 255).astype(np.uint8)

    flipped = np.flip(clipped, 3)
    transposed = flipped.transpose(0, 4, 1, 3, 2)
    reshaped = transposed.reshape(-1, phrases.shape[1] * phrases.shape[4],
                                  phrases.shape[3], phrases.shape[2])

    merged_phrases = []
    phrase_shape = (phrases.shape[4], phrases.shape[1])
    for phrase in reshaped:
        merged_phrases.append(get_image_grid(phrase, phrase_shape, 1,
                                             grid_color))

    merged = get_image_grid(np.stack(merged_phrases), shape, grid_width,
                            grid_color, frame)
    imageio.imwrite(filepath, merged)

def parse_arguments():
    """Parse and return the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', help="Directory to save all the results.")
    parser.add_argument('--params', help="Path to the model parameter file.")
    parser.add_argument('--config', help="Path to the configuration file.")
    parser.add_argument('--gpu', '--gpu_device_num', type=str, default="0",
                        help="The GPU device number to use.")
    parser.add_argument('--n_jobs', type=int,
                        help="Number of parallel calls to use for input "
                             "pipeline. Set to 1 to disable multiprocessing.")
    args = parser.parse_args()
    return args


def setup_dirs(config):
    """Setup an experiment directory structure and update the `params`
    dictionary with the directory paths."""
    # Get experiment directory structure
    config['exp_dir'] = os.path.realpath(config['exp_dir'])
    config['src_dir'] = os.path.join(config['exp_dir'], 'src')
    config['eval_dir'] = os.path.join(config['exp_dir'], 'eval')
    config['model_dir'] = os.path.join(config['exp_dir'], 'model')
    config['sample_dir'] = os.path.join(config['exp_dir'], 'samples')
    config['log_dir'] = os.path.join(config['exp_dir'], 'logs', 'train')

    config['metric_map'] = np.array([
            # indices of tracks for the metrics to compute
            [True] * 8,  # empty bar rate
            [True] * 8,  # number of pitch used
            [False] + [True] * 7,  # qualified note rate
            [False] + [True] * 7,  # polyphonicity
            [False] + [True] * 7,  # in scale rate
            [True] + [False] * 7,  # in drum pattern rate
            [False] + [True] * 7  # number of chroma used
        ], dtype=bool)

    config['tonal_distance_pairs']= [(1, 2)]  # pairs to compute the tonal distance
    config['scale_mask']= list(map(bool, [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]))
    config['drum_filter']= np.tile([1., .1, 0., 0., 0., .1], 16)
    config['tonal_matrix_coefficient']= (1., 1., .5)

    config['track_names'] = (
        'Drums', 'Piano', 'Guitar', 'Bass', 'Ensemble', 'Reed', 'Synth Lead',
        'Synth Pad'
    )
    config['beat_resolution']= 24

    # Make sure directories exist
    for key in ('log_dir', 'model_dir', 'sample_dir', 'src_dir'):
        make_sure_path_exists(config[key])


def _save_pianoroll(array, suffix, name, config, params):
    from musegan.io_utils import image_grid
    filepath = ('./src/'+name)
    # if 'hard_thresholding' in name:
    array = (array > 0) ##二值化

    # # elif 'bernoulli_sampling' in name:
    # rand_num = np.random.uniform(size=array.shape)
    # array = (.5 * (array + 1.) > rand_num)
    print("shape:", np.shape(array))
    save_pianoroll(
        filepath, array, config['midi']['programs'],
        list(map(bool, config['midi']['is_drums'])),
        config['midi']['tempo'], params['beat_resolution'],
        config['midi']['lowest_pitch'])
    return np.array([0], np.int32)

def setup():
    """Parse command line arguments, load model parameters, load configurations,
    setup environment and setup loggers."""
    # Parse the command line arguments
    args = parse_arguments()

    # Load parameters
    params = load_yaml(args.params)
    if params.get('is_accompaniment') and params.get('condition_track_idx') is None:
        raise TypeError("`condition_track_idx` cannot be None type in "
                        "accompaniment mode.")

    # Load configurations
    config = load_yaml(args.config)
    update_not_none(config, vars(args))

    # Set unspecified schedule steps to default values
    for target in (config['learning_rate_schedule'], config['slope_schedule']):
        if target['start'] is None:
            target['start'] = 0
        if target['end'] is None:
            target['end'] = config['steps']

    # Setup experiment directories and update them to configuration dictionary
    setup_dirs(config)

    # Setup loggers
    del logging.getLogger('tensorflow').handlers[0]
    setup_loggers(config['log_dir'])

    # Setup GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu']

    # Backup source code
    backup_src(config['src_dir'])

    return params, config

def load_training_data(params, config):
    """Load and return the training data."""
    # Load data
    if params['is_conditional']:
        raise ValueError("Not supported yet.")
    else:
        labels = None
    LOGGER.info("Loading training data.")
    data = load_data(config['data_source'], config['data_filename'])
    LOGGER.info("Training data size: %d", len(data))

    # Build dataset
    LOGGER.info("Building dataset.")
    print("abcabc", config['batch_size'], params['data_shape'])

    dataset = get_dataset(
        data, labels, NUM_SAMPLE, params['data_shape'],
        config['use_random_transpose'], config['n_jobs'])

    # Create iterator
    if params['is_conditional']:
        train_x, train_y = dataset.make_one_shot_iterator().get_next()
    else:
        train_x, train_y = dataset.make_one_shot_iterator().get_next(), None

    return train_x, train_y

def save_pianoroll(filename, pianoroll, programs, is_drums, tempo,
                   beat_resolution, lowest_pitch):
    """Saves a batched pianoroll array to a npz file."""
    if not np.issubdtype(pianoroll.dtype, np.bool_):
        raise TypeError("Input pianoroll array must have a boolean dtype.")
    if pianoroll.ndim != 5:
        raise ValueError("Input pianoroll array must have 5 dimensions.")
    if pianoroll.shape[-1] != len(programs):
        raise ValueError("Length of `programs` does not match the number of "
                         "tracks for the input array.")
    if pianoroll.shape[-1] != len(is_drums):
        raise ValueError("Length of `is_drums` does not match the number of "
                         "tracks for the input array.")

    reshaped = pianoroll.reshape(
        -1, pianoroll.shape[1] * pianoroll.shape[2], pianoroll.shape[3],
        pianoroll.shape[4])

    # Pad to the correct pitch range and add silence between phrases
    to_pad_pitch_high = 128 - lowest_pitch - pianoroll.shape[3]
    padded = np.pad(
        reshaped, ((0, 0), (0, pianoroll.shape[2]),
                   (lowest_pitch, to_pad_pitch_high), (0, 0)), 'constant')


    # Reshape the batched pianoroll array to a single pianoroll array
    pianoroll_ = padded.reshape(-1, padded.shape[2], padded.shape[3])
    print("pianoroll_", np.shape(pianoroll_))
    # Create the tracks
    tracks = []
    for idx in range(pianoroll_.shape[2]):
        tracks.append(pypianoroll.Track(
            pianoroll_[..., idx], programs[idx], is_drums[idx]))

    # Create and save the multitrack
    multitrack = pypianoroll.Multitrack(
        tracks=tracks, tempo=tempo, beat_resolution=beat_resolution)
    multitrack.save(filename)

def load_or_create_samples(params, config):
    """Load or create the samples used as the sampler inputs."""
    # Load sample_z
    LOGGER.info("Loading sample_z.")
    sample_z_path = os.path.join(config['model_dir'], 'sample_z.npy')
    if os.path.exists(sample_z_path):
        sample_z = np.load(sample_z_path)
        if sample_z.shape[1] != params['latent_dim']:
            LOGGER.info("Loaded sample_z has wrong shape")
            resample = True
        else:
            resample = False
    else:
        LOGGER.info("File for sample_z not found")
        resample = True

    # Draw new sample_z
    if resample:
        LOGGER.info("Drawing new sample_z.")
        sample_z = scipy.stats.truncnorm.rvs(
            -2, 2, size=(np.prod(config['sample_grid']), params['latent_dim']))
        make_sure_path_exists(config['model_dir'])
        np.save(sample_z_path, sample_z)

    if params.get('is_accompaniment'):
        # Load sample_x
        LOGGER.info("Loading sample_x.")
        sample_x_path = os.path.join(config['model_dir'], 'sample_x.npy')
        if os.path.exists(sample_x_path):
            sample_x = np.load(sample_x_path)
            if sample_x.shape[1:] != params['data_shape']:
                LOGGER.info("Loaded sample_x has wrong shape")
                resample = True
            else:
                resample = False
        else:
            LOGGER.info("File for sample_x not found")
            resample = True

        # Draw new sample_x
        if resample:
            LOGGER.info("Drawing new sample_x.")
            data = load_data(config['data_source'], config['data_filename'])
            sample_x = get_samples(
                np.prod(config['sample_grid']), data,
                use_random_transpose = config['use_random_transpose'])
            make_sure_path_exists(config['model_dir'])
            np.save(sample_x_path, sample_x)
    else:
        sample_x = None

    return sample_x, None, sample_z

from keras.utils import to_categorical

def frisky(output):  ##
    # elif 'bernoulli_sampling' in name:
    # print(output.shape)
    # print(output[0][10])
    # print(output[0][20])
    # print(output[0][30])
    rand_num = np.random.uniform(size=output.shape)
    #output = (.5 * (output + 1.) > rand_num)
    # print("111111111")
    output[output > 0] = 1
    output[output < 0] = -1
    return output

def frisky2(output):  ##
    print("22222222")
    dim = output.shape[-1]
    shape1 = output.shape[0]
    shape2 = output.shape[1]
    num = output.shape[0] * output.shape[1]
    output = np.reshape(output, [output.shape[0] * output.shape[1], dim])
    input_matrix = np.zeros((num, dim), dtype=np.int)
    # load_data(output.numpy())
    # preds = output.numpy()
    # probas = np.random.multinomial(1,preds[index],1)
    # probas = np.random.choice(np.arange(0,128),p=output[index+1].numpy())
    for index in range(num):
        probas = np.argmax(output[index])
        # input_matrix = tf.reshape(input_matrix, [num, 128]).numpy()
        if(probas == 0):
            input_matrix[index] = np.zeros((dim), dtype=np.int)
        else:
            input_matrix[index] = to_categorical(probas, dim)
        # input_matrix[index+1] = probas
    input_matrix = np.reshape(input_matrix, [shape1, shape2, dim])
    return input_matrix

def trans2norm(tests1):
    tests1 = np.transpose(tests1, (1, 2, 3, 0))
    tests1 = np.resize(tests1, (tests1.shape[0] // 4, 4, tests1.shape[1], tests1.shape[2], tests1.shape[3]))
    return tests1

def print_metrix(sess, data, params, config, string = "", flag = 0):
    ## data should (?, 4, long, pitch, track)


    print("**********************************"+string)
    eval_data = data
    if flag == 0:
        eval_data = tf.convert_to_tensor(trans2norm(data))

    binarized = tf.round(.5 * (eval_data + 1.))  ## 从-1 到 1 ？ 数值取整
    save_metric_ops = get_save_metric_ops(
        binarized, params['beat_resolution'], 0,
        config['eval_dir'])
    # save_metric_ops = get_metric_ops(binarized, params['beat_resolution'])
    sess.run(save_metric_ops)

    print("**********************************"+string)


def save2midi(song, name, config, params):
    song = song
    _save_pianoroll(array=song, suffix=None, name=name, config=config, params=params)

    m = Multitrack('./src/'+name+'.npz')
    m.write('./src/'+name+'.mid')

def main():
    """Main function."""
    # Setup

    logging.basicConfig(level=LOGLEVEL, format=LOG_FORMAT)
    params, config = setup()
    LOGGER.info("Using parameters:\n%s", pformat(params))
    LOGGER.info("Using configurations:\n%s", pformat(config))

    ## 记录 history
    history = LossHistory()

    # ================================== Data ==================================

    # Load training data
    m_train_xt, _ = load_training_data(params, config)


    ### for binary_crossentropy
    # m_train_x += 1
    # m_train_x *= 0.5
    ###

    print("TRUE TRAIN SHAPE IS", np.shape(m_train_xt))
    ##转换为数组
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # 转化为numpy数组
    # 通过.eval函数可以把tensor转化为numpy类数据
    print(type(m_train_xt))

    m_train_x = m_train_xt.eval(session=sess)

    from data_proc import init_stage1_data, init_stage2_data
    train_x = init_stage1_data(m_train_x)

    ###################### test eval
    # metrix_data = train_x
    #
    # print(np.shape(metrix_data))
    # print_metrix(sess = sess, data=metrix_data, params=params, config=config)

    ######################
    # train_x = np.array(train_x)
    # m_train_x = m_train_x[0:32]
    # save2midi(m_train_x, "ycn", config, params)
    # print(np.shape(train_x))
    # train_x = np.transpose(train_x, (1, 2, 3, 0))
    # print(np.shape(train_x))
    # train_x = train_x[0:64]
    # train_x = np.reshape(train_x, (train_x.shape[0] // 4, 4, train_x.shape[1], train_x.shape[2], train_x.shape[3]))
    # save2midi(train_x, "lxxaaa", config, params)
    # exit(0)
    len_track = len(train_x)
    track_vae = []

    for i in range(len_track):
        input_dim = int(train_x[i].shape[2])
        timesteps = int(train_x[i].shape[1])
        print("input dim", input_dim, "timesteps", timesteps)
        batch_size = 32
        latent_dim = 256
        vrae = vrae_muse(input_dim,
                     timesteps=timesteps,
                     batch_size=batch_size,
                     intermediate_dim=256,
                     latent_dim=latent_dim,
                     epsilon_std=0.01)

        track_vae.append(vrae)

    print("lets training!!!")
    spli = batch_size * ((NUM_SAMPLE//batch_size))

    #stage1: 训练各个track
    if TRAIN_STAGE1:
        print("stage 1:")
        for i in range(len_track):
            x = train_x[i][0:spli]
            print("NO",str(i),np.shape(x))
            track_vae[i].vae.fit(x, x, epochs=100, callbacks=[history], validation_split=0.1)
            track_vae[i].vae.save_weights(DATA_STRING+str(i) + "track_vae.h5")
            track_vae[i].encoder.save_weights(DATA_STRING+str(i) + "track_enc.h5")
            track_vae[i].generator.save_weights(DATA_STRING+str(i) + "track_gen.h5")

            history.loss_plot('epoch', "stage1")

    else:
        for i in range(len_track):
            print("loading track vae..")
            track_vae[i].vae.load_weights(DATA_STRINT+str(i)+"track_vae.h5")
            track_vae[i].encoder.load_weights(DATA_STRINT+str(i)+"track_enc.h5")
            track_vae[i].generator.load_weights(DATA_STRINT+str(i)+"track_gen.h5")

    ##测试二值化网络
    if BINARY_OUTPUT == True:
        ##创建一个二值化网络，把之前的加进去并且训练
        print("ADD BINARY TO STAGE 1:")
        ##创建 BINARY模型
        train_bns = []
        bn_refines = []
        flag = 2 ##1是以bar为单位 2是以列为单位
        flag_loss = 1## 1是MSE，2是bar的b cross, 3是列的b cross

        for i in range(len_track):
            bn = BN_REFINE(input_dim=input_dim,
                           timesteps=timesteps,
                           num_track=1,
                           flag= flag,
                           loss_flag = flag_loss)

            if BN_REFINE_TRAIN == False:
                if flag == 1:
                    bn.model.load_weights(str(flag_loss)+DATA_STRING + str(i) + "bn.h5")
                else:
                    bn.model.load_weights(DATA_STRING + str(i) + "bn.h5")
            else:
                x = train_x[i][0:spli] ##元数据
                train_bn = track_vae[i].vae.predict(x) ##生成的数据
                print("bn_x shape", np.shape(train_bn))
                print("bn_y shape", np.shape(x))
                ##创建一个BN_REFINE网络

                x += 1
                x *= 0.5

                if flag == 1:
                    x = np.expand_dims(x, axis=-1)
                    train_bn = np.expand_dims(train_bn, axis=-1)
                else:
                    x = np.reshape(x, newshape= [x.shape[0]*x.shape[1], x.shape[2]])
                    train_bn = np.reshape(train_bn, newshape=[train_bn.shape[0] * train_bn.shape[1], train_bn.shape[2]])

                bn.model.fit(train_bn, x, batch_size = 32, epochs = 100, callbacks=[history], validation_split=0.1)
                if flag == 1:
                    bn.model.save_weights(str(flag_loss)+DATA_STRING + str(i) + "bn.h5")
                else:
                    bn.model.save_weights(DATA_STRING + str(i) + "bn.h5")
                history.loss_plot('epoch', "bns")
            bn_refines.append(bn)

        print(np.shape(train_bns))

    ####测试第一阶段的生成
    if GEN_STAGE1 == True:
        print("Generator Stage 1")

        pres = []
        pres_bns = []
        tests1 = []
        tests2 = []
        for i in range(len_track):
            test_x = train_x[i][0:32]
            pre = track_vae[i].vae.predict(test_x) #直接

            if BINARY_OUTPUT == True:
                #####################################################################
                #### BN层的输出精炼
                press = pre ## BN的输入
                ta, tb, tc = press.shape[0], press.shape[1], press.shape[2]
                if flag == 2:
                    press = np.reshape(press, newshape=[ta * tb, tc]) ##BN输入转换
                else:
                    press = np.expand_dims(press, axis=-1)
                pre_bn = bn_refines[i].model.predict(press) ##送给BN 输出
                pre_bn[pre_bn >= 0.5] = 1
                pre_bn[pre_bn < 0.5] = -1
                pre_bn = np.reshape(pre_bn, newshape=[ta, tb, tc])
                pre_bn = frisky(pre_bn)
                pres_bns.append(pre_bn)
                #######################################################################

            pre = frisky(pre)
            pres.append(pre)


            tests1.append(test_x)
            test2_x = frisky(test_x)
            tests2.append(test2_x)
        #     print(" ")
        # exit(0)

        print(np.shape(pres))


        metrix_data = train_x

        print(np.shape(metrix_data))
        print_metrix(sess = sess, data=metrix_data, params=params, config=config)

        print_metrix(sess, tests1, params, config, "tests111")
        tests1 = np.transpose(tests1, (1, 2, 3, 0))
        tests1 = np.resize(tests1, (tests1.shape[0] // 4, 4, tests1.shape[1], tests1.shape[2], tests1.shape[3]))
        shape = [1, 8]
        save_image("./src/test1.png", tests1, shape)

        print("tests shape", np.shape(tests1))
        save2midi(tests1, name="test1", config=config, params=params)


        tests2 = np.transpose(tests2, (1, 2, 3, 0))
        tests2 = np.resize(tests2, (tests2.shape[0] // 4, 4, tests2.shape[1], tests2.shape[2], tests2.shape[3]))
        print("tests shape", np.shape(tests2))
        save2midi(tests2, name="test2_f", config=config, params=params)
        save_image("./src/test2_f.png", tests2, shape)


        print_metrix(sess, pres, params, config, "preee111")
        song = np.transpose(pres, (1, 2, 3, 0))
        song = np.resize(song, (song.shape[0] // 4, 4, song.shape[1], song.shape[2], song.shape[3]))
        print("song shape", np.shape(song))
        save2midi(song, name="test3_pre", config=config, params=params)
        save_image("./src/test3_pre.png", song, shape)

        if BINARY_OUTPUT == True:
            print_metrix(sess, pres_bns, params, config, "preee222_bns")
            song = np.transpose(pres_bns, (1, 2, 3, 0))
            song = np.resize(song, (song.shape[0] // 4, 4, song.shape[1], song.shape[2], song.shape[3]))
            print("song shape", np.shape(song))
            save2midi(song, name="test4_prebn", config=config, params=params)
            save_image("./src/test4_prebn.png", song, shape)

        ##########################随机生成

        print("Random Generator Test Stage1")
        loop = 32
        song = []
        for i in range(len_track):
            z = []
            for j in range(loop):
                z.append(np.random.normal(loc=0.0, scale=0.1, size=(latent_dim)))
            z = np.reshape(z, newshape=(loop, latent_dim))
            print(np.shape(z))
            t = track_vae[i].generator.predict(z)  # latent: z, z_sigma, z_log
            t = frisky(t)
            song.append(t)

        print_metrix(sess, song, params, config, "preee222_random")
        song = np.transpose(song, (1, 2, 3, 0))
        song = np.reshape(song, (song.shape[0] // 4, 4, song.shape[1], song.shape[2], song.shape[3]))
        print("song shape", np.shape(song))
        save2midi(song, name="test_random", config=config, params=params)

    # stage2: 训练多模态str(i)融合与生成
    # 就是采样隐向量，然后直接，和那个一样。

    print("stage 2:")
    latent_list = []
    for i in range(len_track):
        x = train_x[i][0:spli]
        print(np.shape(x))
        latent = track_vae[i].encoder.predict(x) #latent: z, z_sigma, z_log
        print(np.shape(latent))
        latent_list.append(latent)

    print("latent list shape", np.shape(latent_list))

    #然后处理作为中间VAE的输入。
    train2_x = init_stage2_data(latent_list)
    print("train2 shape", np.shape(train2_x))

    input_dim = int(train2_x.shape[2])
    timesteps = track_num = int(train2_x.shape[1])

    batch_size = 32
    latent_dim2 = 64
    ## 使用VRAE
    vrae = vrae_muse(input_dim,
                     timesteps=timesteps,
                     batch_size=batch_size,
                     intermediate_dim=128,
                     latent_dim=latent_dim2,
                     epsilon_std=0.01)

    # # ##使用DNN来融合
    #
    # vrae = vrae_fusion(input_dim,
    #                    batch_size,
    #                    track_num,
    #                    latent_dim2,
    #                    epsilon_std=1.)
    # train2_x = input_trans(train2_x)

    if(TRAIN_STAGE2 == True):
        vrae.vae.fit(train2_x, train2_x, epochs= 500, batch_size = 32, callbacks = [history], validation_split = 0.1)
        vrae.vae.save_weights(DATA_STRING+"m_vae.h5")
        vrae.encoder.save_weights(DATA_STRING+"m_enc.h5")
        vrae.generator.save_weights(DATA_STRING+"m_gen.h5")
        history.loss_plot('epoch', "stage2")
    else:
        vrae.vae.load_weights(DATA_STRING+"m_vae.h5")
        vrae.encoder.load_weights(DATA_STRING+"m_enc.h5")
        vrae.generator.load_weights(DATA_STRING+"m_gen.h5")

        # vrae.vae.fit(train2_x, train2_x, epochs=500, batch_size=32)
        #
        # vrae.vae.save_weights(DATA_STRING + "m_vae.h5")
        # vrae.encoder.save_weights(DATA_STRING + "m_enc.h5")
        # vrae.generator.save_weights(DATA_STRING + "m_gen.h5")

    # #最后通过encoder来完成生成歌曲，并且试听。


    ##先看看还原的情况如何：然后再看生成的情况:
    song_org = []
    song_re = []
    org_x = train2_x[0:32]
    rebuild_x = vrae.vae.predict(org_x)

    for i in range(len_track):
        print("Org Generator Test Stage2", str(i))
        z_mean_org = org_x[:,i,0:latent_dim]
        z_log_org = org_x[:, i, latent_dim:]

        z_mean_re = rebuild_x[:,i,0:latent_dim]
        z_log_re = rebuild_x[:, i, latent_dim:]
        z_err = np.random.normal(loc=0.0, scale=0.01, size=z_mean_org.shape)

        z_org = z_mean_org+z_log_org * z_err
        z_re = z_mean_re+z_log_re * z_err


        t_org = track_vae[i].generator.predict(z_org)  # latent: z, z_sigma, z_log
        t_org = frisky(t_org)

        t_re = track_vae[i].generator.predict(z_re)  # latent: z, z_sigma, z_log
        t_re = frisky(t_re)
        song_org.append(t_org)
        song_re.append(t_re)

    print_metrix(sess, song_org, params, config, "song_org")
    song_org = np.transpose(song_org, (1,2,3,0))
    song_org = np.reshape(song_org, (song_org.shape[0]//4, 4, song_org.shape[1], song_org.shape[2], song_org.shape[3]))
    print("song_org shape", np.shape(song_org))
    save2midi(song_org, name="song_org_test", config = config, params=params)

    print_metrix(sess, song_re, params, config, "song_re_test")
    song_re = np.transpose(song_re, (1, 2, 3, 0))
    song_re = np.reshape(song_re, (song_re.shape[0] // 4, 4, song_re.shape[1], song_re.shape[2], song_re.shape[3]))
    print("song_re shape", np.shape(song_re))
    save2midi(song_re, name="song_re_test", config=config, params=params)


    ## 随机生成来来来
    loop = 32
    z_gen = []
    for i in range(loop):
        z_gen.append(np.random.normal(loc=0.0, scale=0.01, size=(latent_dim2)))
    z_gen = np.reshape(z_gen, newshape=[loop, latent_dim2])
    gens = vrae.generator.predict(z_gen)

    # gens = output_trans(gens, track_num)

    song = []
    song_bns = []
    for i in range(len_track):
        if TEST_STAGE2 == True:
            print("Random Generator Test Stage2")
            z_mean = gens[:,i,0:latent_dim]
            z_log = gens[:, i, latent_dim:]
            z = np.random.normal(loc=0.0, scale=0.01, size=z_mean.shape)
            z = z_mean+z_log*z

        print(np.shape(z))
        t = track_vae[i].generator.predict(z)  # latent: z, z_sigma, z_log

        #####################################################################
        #### BN层的输出精炼
        if BINARY_OUTPUT == True:
            press = t  ## BN的输入
            ta, tb, tc = press.shape[0], press.shape[1], press.shape[2]
            press = np.reshape(press, newshape=[ta * tb, tc])  ##BN输入转换
            song_bn = bn_refines[i].model.predict(press)  ##送给BN 输出
            song_bn[song_bn >= 0.5] = 1
            song_bn[song_bn < 0.5] = -1
            song_bn = np.reshape(song_bn, newshape=[ta, tb, tc])
            song_bn = frisky(song_bn)
            song_bns.append(song_bn)
        #######################################################################

        t = frisky(t)
        song.append(t)

    song = np.transpose(song, (1,2,3,0))
    song = np.reshape(song, (song.shape[0]//4, 4, song.shape[1], song.shape[2], song.shape[3]))
    print("song shape", np.shape(song))
    save2midi(song, name="rnn_song_random", config = config, params=params)

    if BINARY_OUTPUT == True:
        song_bns = np.transpose(song_bns, (1, 2, 3, 0))
        song_bns = np.reshape(song_bns, (song_bns.shape[0] // 4, 4, song_bns.shape[1], song_bns.shape[2], song_bns.shape[3]))
        print("song_bns shape", np.shape(song_bns))
        save2midi(song_bns, name="rnn_song_bns_random", config=config, params=params)



    LOGGER.info("Training end")


if __name__ == "__main__":
    main()
