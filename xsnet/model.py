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


class XSNet(Chain):
    def __init__(self, n_layers, n_source_pose_node, n_target_rhythm, n_units):
        super(XSNet, self).__init__()
        with self.init_scope():
            self.embed_x = L.EmbedID(n_source_pose_node, n_units)
            self.embed_y = L.EmbedID(n_target_rhythm, n_units)
            self.encoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
            self.decoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
            self.W = L.Linear(n_units, n_target_rhythm)

        self.n_layers = n_layers
        self.n_units = n_units

    def __call__(self, xs, ys):
        hx, cx, _ = self.encoder(None, None, xs)
        _, _, os = self.decoder(hx, cx, ys)
        
        pass
