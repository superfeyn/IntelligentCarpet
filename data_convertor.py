import h5py
import os
import pandas as pd
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate
import random
import re

def load_pressure_data(path):
    cls_idx = dict(back=0, forward=1, left=2, right=3, stand=4)
    datas, speeds, angles, classes = [], [], [], []
    files = os.listdir(path)
    for file in files:
    #for i in range(1):
        filename = os.path.join(path, file)
        with h5py.File(filename, "r") as f:
            # Get the data
            data = np.array(list(f["pressure"]))
            length = len(data)
            cls, r_speed, r_angle = re.split(r'[_ _]', filename[:-4])
            r_cls = cls_idx[cls]
            if cls == 'stand':
                r_angle = 0
                r_speed = 0
            speed = np.full((length, 1), r_speed)
            angle = np.full((length, 1), r_angle)
            cls = np.full((length, 1), r_cls)
        datas.append(data)
        speeds.append(speed)
        angles.append(angle)
        classes.append(cls)
        print(f"loaded {filename}")
    return np.concatenate(datas), np.concatenate(speeds), np.concatenate(angles), np.concatenate(classes)

def save_data_and_label(data, label, size, path, id_string=''):
    print(f"saving to {path}{id_string}")
    for i in range(0, len(data), size):
        data_dict = {
            "data": data[i:i+size],
            "label": label[i:i+size],

        }
        with open(path + id_string + f'{i+size}.pickle', 'wb') as f:
            pickle.dump(data_dict, f)

def save_data_and_label_sliding_window(data, label, save_size, window_size, path, id_string=''):
    print(f"saving to {path}{id_string}")
    datas = []
    labels = []
    i = 0
    length = len(data)
    while i+window_size < length:
        datas.append([data[i:i+window_size]])
        labels.append([label[i:i+window_size]])
        if i % save_size == 0 and i != 0:
            data_dict = {
                "data": np.concatenate(datas),
                "label": np.concatenate(labels)
            }
            with open(path + id_string + f'{i+save_size}.pickle', 'wb') as f:
                pickle.dump(data_dict, f)
            datas = []
            labels = []
        i += 1

def augment_pressure(origin_image, angle, x, y):
    foot_size = 32
    foot_img = origin_image[16:48, 16:48]
    new_img = np.random.uniform(0, 0.001, origin_image.shape)
    foot_img = rotate(foot_img, angle=angle, reshape=False)
    new_img[x:x+foot_size, y:y+foot_size] = foot_img
    return new_img
 
def augment_data_label(data, speed, angle):
    foot_size = 32
    angle_value = random.randint(0, 360)
    x, y = random.randint(0, len(data[0])-1 - foot_size), random.randint(0, len(data[0])-1 - foot_size)

    for i in range(len(data)):
        angle[i] = (angle[i]+angle_value)%360
        data[i] = augment_pressure(data[i], angle_value, x, y)
    return data, speed, angle

def sliding_windows(array, window_size):
    total_data = []
    for i in range(len(array) - window_size):
        total_data.append([array[i:i+window_size]])
    return np.concatenate(total_data)

def show_pressure(img):
    plt.imshow(img, interpolation='nearest')
    plt.show()

if __name__ == "__main__":
    path = "./walking_data/train/"
    path_new = "./walking_data/train/augmented/"
    augment_num = 1
    window_size = 20
    save_size = 100
    data, speed, angle, classes = load_pressure_data(path)

    for i in range(augment_num):
        data, speed, angle, classes = augment_data_label(data, speed, angle)
        label = np.concatenate((speed, angle, classes))

        save_data_and_label_sliding_window(data, label, save_size, window_size, path_new, id_string=f"{i}_")

        #data_window, label_window = sliding_windows(data, window_size), sliding_windows(label, window_size)
        #save_data_and_label(data_window, label_window, save_size, path_new, id_string=f"{i}_")
    
    



