# Copyright 2018 PikachuHy. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Prepare neural network training data sets
For video, use openpose to extract body keypoints
For Audio, use midi file to construct data
"""
import numpy as np
from chainer.datasets import tuple_dataset


def get_tuple(data):
    data = np.array(data)
    print('get_tuple data:',data.shape)
    t_data = []
    t_label = []
    for it in data:
        t_data.append(np.array(it[0]))
        t_label.append(np.array(it[1]))
    t_data = np.array(t_data)
    t_label = np.array(t_label)
    print("t_data:",t_data.shape)
    print("t_data[0].shape:",t_data[0].shape)
    return tuple_dataset.TupleDataset(t_data, t_label)


def get_data():
    DATASET_PATH = '/home/pikachu/Documents/pose.npz'
    train = []
    test = []
    data = np.load(DATASET_PATH)
    data = data['arr_0']
    data = np.array(data)
    print('type num:', len(data), data.shape)
    for it in data:
        it = np.array(it)
        print('video num:', len(it))
        # Use 90% for the training set and 10% for the test set
        n = len(it) * 0.9
        n = int(n)
        t_train = it[:n]
        t_test = it[n + 1:]
        train.extend(t_train)
        test.extend(t_test)

    train = np.array(train)
    test = np.array(test)
    print(len(train), len(test),train.shape,test.shape)
    print(len(train[0]))
    train = get_tuple(train)
    test = get_tuple(test)
    return train,test


if __name__ == '__main__':
    get_data()
