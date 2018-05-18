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
Define a neural network that converts the rhythm of motion into music
Based on information already available define a neural network similar to Seq2Seq
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
from chainer.functions.loss import softmax_cross_entropy
from chainer.functions.evaluation import accuracy
from chainer import reporter


def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs


class XSNet(Chain):
    def __init__(self, n_layers, n_source_pose_node, n_target_rhythm, n_units):
        super(XSNet, self).__init__()
        with self.init_scope():
            self.embed_x = L.Linear(n_source_pose_node, n_units)
            self.embed_y = L.EmbedID(n_target_rhythm, n_units)
            self.encoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
            self.decoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
            self.W = L.Linear(n_units, n_target_rhythm)

        self.n_layers = n_layers
        self.n_units = n_units

    def __call__(self, xs, ys):
        eys = sequence_embed(self.embed_y, ys)
        xs = [x[::-1] for x in xs]
        # xs = np.array(xs)
        # exs = self.embed_x(xs)
        # exs = [self.embed_x(it) for it in xs]
        exs = []
        for it in xs:
            t = self.embed_x(it)
            exs.append(t)
        hx, cx, _ = self.encoder(None, None, exs)
        _, _, os = self.decoder(hx, cx, eys)
        batch = len(xs)
        concat_os = F.concat(os, axis=0)
        concat_ys_out = F.concat(ys, axis=0)
        loss = F.sum(F.softmax_cross_entropy(self.W(concat_os), concat_ys_out, reduce='no')) / batch
        chainer.report({'loss': loss.data}, self)

        n_words = concat_ys_out.shape[0]
        perp = self.xp.exp(loss.data * batch / n_words)
        chainer.report({'perp': perp}, self)
        return loss


class Classifier(Chain):
    compute_accuracy = True

    def __init__(self, predictor,
                 lossfun=softmax_cross_entropy.softmax_cross_entropy,
                 accfun=accuracy.accuracy,
                 label_key=-1):
        if not (isinstance(label_key, (int, str))):
            raise TypeError('label_key must be int or str, but is %s' %
                            type(label_key))

        super(Classifier, self).__init__()
        self.lossfun = lossfun
        self.accfun = accfun
        self.y = None
        self.loss = None
        self.accuracy = None
        self.label_key = label_key

        with self.init_scope():
            self.predictor = predictor

    def __call__(self, *args, **kwargs):
        self.y = None
        self.loss = None
        self.accuracy = None
        self.y = self.predictor(*args, **kwargs)
        self.loss = self.lossfun(self.y, t)
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            self.accuracy = self.accfun(self.y, t)
            reporter.report({'accuracy': self.accuracy}, self)
        return self.loss
        # 子晋的代码
        # x = self.predictor(x, t)
        # label = np.transpose(t)[0]
        # cross_entropy = F.mean(F.softmax_cross_entropy(x, label))
        # accuracy = F.accuracy(x, label)
        # chainer.report({'loss': cross_entropy, 'accuracy': accuracy}, self)
        # return cross_entropy
