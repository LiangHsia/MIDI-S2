import numpy as np

def init_tong_stage1_data(data):
    data = np.transpose(data, [0, 1, 4, 2, 3])
    b = np.reshape(data, newshape=[data.shape[0]*data.shape[1]*data.shape[2], data.shape[3], data.shape[4]])
    return b

def init_stage1_data(data):
    track1 = data[:,:,:,:,0]
    track2 = data[:,:,:,:,1]
    track3 = data[:,:,:,:,2]
    track4 = data[:, :, :, :, 3]
    track5 = data[:, :, :, :, 4]

    def fun4(t):
        print(t.shape[0], t.shape[1], t.shape[2], t.shape[3])
        t = np.resize(t, new_shape=[t.shape[1]*t.shape[0],t.shape[2],t.shape[3]])
        return t

    def fun(t):
        print(t.shape[0], t.shape[1], t.shape[2], t.shape[3])
        b = np.reshape(t, newshape=[t.shape[1]*t.shape[0], t.shape[2], t.shape[3]])
        return b

    def funf(t):
        return t

    data_list = []
    data_list.append(fun(track1))
    data_list.append(fun(track2))
    data_list.append(fun(track3))
    data_list.append(fun(track4))
    data_list.append(fun(track5))
    return data_list

def init_stage2_data(data): #(track, z/log/sigma, num, latent_size)
    print(np.shape(data))
    len_data = len(data)
    print("len",len_data)

    train2_x = []
    print(np.shape(data))
    t1 = np.array(data)[:,1,:,:] #mean
    t2 = np.array(data)[:,2,:,:] #z_log_sigma
    t3 = np.array(data)[:,0,:,:] #z
    t = np.concatenate((t1, t2), axis=2)
    print("z",np.shape(t))
    t = np.transpose(t, (1, 0, 2))
    print("Transpose",np.shape(t))
    #t = np.reshape(t, newshape=(t.shape[0], t.shape[1]*t.shape[2]))
    print("over", np.shape(t))
    return t

