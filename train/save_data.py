import pickle
import h5py
import numpy as np
import os

def load_data(path, save_train_dir, save_val_dir, percent):
    files = os.listdir(path)
    tfolder_num = 0
    vfolder_num = 0
    data_tlog = []
    data_vlog =[]
    for file in files:
        filename = os.path.join(path, file)
        data = pickle.load(open(filename, "rb"))
        tactile = data["data"]
        speed = data["label"][:,0]
        angle = data["label"][:,1]
        cls= data["label"][:,2]

        for i in range(0, int(tactile.__len__()/100*percent)):
            if (tfolder_num % 10000 == 0):
                data_tlog.append(tfolder_num)
                os.mkdir(save_train_dir + str((tfolder_num // 10000) * 10000))
            d = [tactile[i], np.array([np.float64(speed[i])/170., np.float64(angle[i])/360.]).astype(np.float64), cls[i].astype(np.int64)]
            pickle.dump(d, open(save_train_dir + str(tfolder_num // 10000 * 10000) + "/" + str(tfolder_num) + '.p', "wb"))
            tfolder_num += 1
        for i in range(int(tactile.__len__()/100*percent), int(tactile.__len__())):
            if (vfolder_num % 10000 == 0):
                data_vlog.append(vfolder_num)
                os.mkdir(save_val_dir + str((vfolder_num // 10000) * 10000))
            d = [tactile[i], np.array([np.float64(speed[i])/170., np.float64(angle[i])/360.]).astype(np.float64), cls[i].astype(np.int64)]
            pickle.dump(d, open(save_val_dir + str(vfolder_num // 10000 * 10000) + "/" + str(vfolder_num) + '.p', "wb"))
            vfolder_num += 1

    pickle.dump(data_tlog, open(save_train_dir + 'log.p', "wb"))
    pickle.dump(data_vlog, open(save_val_dir + 'log.p', "wb"))

if __name__ == "__main__":
    path = "/home/shlee/aaSSD/MIT_data/walking_data/cyh_augmented/"
    save_train_dir = "/home/shlee/aaSSD/MIT_data/walking_data/agumented/train/"
    save_val_dir = "/home/shlee/aaSSD/MIT_data/walking_data/agumented/val/"
    load_data(path, save_train_dir, save_val_dir, percent=80)