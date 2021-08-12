import pickle
import h5py
import numpy as np
import os

#log_ = pickle.load(open("/home/shlee/aaSSD/MIT_data/test1/log.p", "rb"))

save_dir = "/home/shlee/aaSSD/MIT_data/val_speed_norm/"
load_dir = "/home/shlee/aaSSD/MIT_data/val_hdf5/"

data_log = []
num = 0
for k in [50,105,170,185,240]:
    filename = load_dir +str(k)+".hdf5"
    #filename = "/home/shlee/aaSSD/MIT_data/train/speed1.hdf5"
    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data = list(f[a_group_key])
        pressure_data = f["pressure"]
        for i in range(0, pressure_data.__len__()):
            if (num % 200 == 0):
                data_log.append(num)
                os.mkdir(save_dir + str(num // 200 * 200))

            d = [f["pressure"][i], np.array([(k-50)/(240-50), 0.5])]
            pickle.dump(d, open(save_dir + str(num // 200 * 200) + "/" + str(num) + '.p', "wb"))
            num+=1

        pickle.dump(data_log, open(save_dir + 'log.p', "wb"))

#0.p : [ndarray(64,64), ndarray(1,2)]
print(data_log)