# -*- coding: UTF-8 -*-
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
A simple test on xsnet
"""
import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import argparse
from model import XSNet, Classifier
import datasets
from chainer.dataset import concat_examples
from chainer.backends.cuda import to_cpu

from chainer.functions.array import transpose_sequence
from PadIterator import PadIterator


def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs


def main():
    train, test = datasets.get_data()
    train_iter = chainer.iterators.SerialIterator(train, 10)
    # train_iter = PadIterator(train, 10)
    embed_x = L.EmbedID(54, 1000)
    train_data = []
    for it in train:
        train_data.append(it[0])
    train_data = np.array(train_data)
    print(train_data.shape)
    print(train_data[0].shape)
    # a = embed_x(train_data[0])
    # print(a)
    n_layers = 3
    n_units = 10
    encoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
    decoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
    embed_y = L.EmbedID(10, n_units)
    param = []
    for it in train_data:
        # print(len(it))
        pass
    # b = encoder(None,None,train_data.tolist())
    # print(b)
    batch = train_iter.next()
    batch = train
    batch_data = []
    batch_label = []
    for it in batch:
        batch_data.append(it[0])
        batch_label.append(it[1])
    # Find the longest
    max_len = 0
    for it in batch_data:
        if len(it) > max_len:
            max_len = len(it)
    print('max_len', max_len)
    # Make each list the same length
    batch_data_2 = []
    batch_label_2 = []
    for it in batch_data:
        n = max_len - len(it)
        arr = np.zeros((n, 54))
        # print('before arr.shape:',arr.shape)
        # print('before it.shape: ', it.shape)
        if it.shape[0] == 0:
            print('it:', it)
            it = arr
            batch_data_2.append(arr)
            continue
        arr = np.concatenate((it, arr), axis=0)
        # print('arr.shape:',arr.shape)
        # print('it.shape: ', it.shape)
        batch_data_2.append(arr)
    for it in batch_label:
        n = max_len - len(it)
        arr = np.zeros((n,))
        print('before arr.shape:', arr.shape)
        print('before it.shape: ', it.shape)
        if it.shape[0] == 0:
            print('it:', it)
            it = arr
            batch_label_2.append(arr)
            continue
        arr = np.concatenate((it, arr), axis=0)
        # print('arr.shape:',arr.shape)
        # print('it.shape: ', it.shape)
        batch_label_2.append(arr)

    # c = transpose_sequence.transpose_sequence(batch_data_2)
    # print(c)
    print('len batch_data_2:', len(batch_data_2))
    print('len batch_data_2[0]:', len(batch_data_2[0]))
    print('len batch_data_2[0][0]:', len(batch_data_2[0][0]))
    # l_embed_x = L.Linear(len(batch_data_2[0]),1000)
    # d = l_embed_x(np.array(batch_data_2))
    d_orgin = np.array(batch_data_2).astype(np.float32)
    t_orgin = np.array(batch_label_2).astype(np.int32)
    dd = []
    # for it in d:
    #     d = encoder(None,None,(it,))
    #     print(d)
    #     dd.append(d)
    l = L.Linear(54, n_units)
    decoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
    for index in range(2):
        d = d_orgin[index]
        print('shape:', d.shape)
        d = l(d)
        print('l shape:', d.shape)
        hx, cx, _ = encoder(None, None, (d,))
        print('encoder shape:', d[0].shape)
        # d = batch_label_2[index]
        d = t_orgin[index]
        print(d)
        print('d shape:', d.shape)
        eys = sequence_embed(embed_y, t_orgin)
        d = decoder(hx, cx, eys)
        print('decoder shape:', d.shape)
        dd.append(d)
        print(d)
    dd = np.array(dd)
    print(dd.shape)
    d3 = []
    # for it in dd:
    #     d = encoder(None,None,(it,))
    #     print('encoder shape:',d[0].shape)
    #     d3.append(d)
    d3 = np.array(d3)
    print('d3 shape:', d3.shape)
    # d = encoder(None,None,(d[0],))
    # print(d)
    # print(d[0].shape)


if __name__ == '__main__':
    main()
