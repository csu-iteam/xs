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
Basic settings of Training the neural network
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
def load_midi_snippet(path):
    with open(path) as f:
        midi_snippets = {line.strip(): i + 1 for i,line in enumerate(f)}
        midi_snippets['0'] = 0
    return midi_snippets
def handle_data(data):
    for i in range(len(data)):
        for j in range(len(data[i][1])):
            data[i][1][j] = data[i][1][j] + 1

    return data
def convert(batch, device):
    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = np.cumsum([len(x)
                                     for x in batch[:-1]], dtype=np.int32)
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    return {'xs': to_device_batch([x for x, _ in batch]),
            'ys': to_device_batch([y for _, y in batch])}

def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1024,
                        help='Number of units')
    parser.add_argument('--layer', '-l', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    parser.add_argument('--target_midi','-t', default='../midi/database.txt',
                        help='target midi snippet')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')


    # Load the MNIST dataset
    train, test = datasets.get_data()
    print('train', len(train))
    # exit(0)
    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    # 统计音乐片段
    target_midi_ids = load_midi_snippet(args.target_midi)
    target_midi_ids = {i:w for w, i in target_midi_ids.items()}
    # 训练集打的标签，下标是从0开始的，在这里，我把0做为空,所以所有的标签需要+1
    train = handle_data(train)
    test = handle_data(test)
    n_rhythm = len(target_midi_ids)
    model = Classifier(XSNet(args.layer, 54, n_rhythm, args.unit))
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, converter=convert, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    # trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    # trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot for each specified epoch
    # frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    # trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    # trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    # if args.plot and extensions.PlotReport.available():
    #     trainer.extend(
    #         extensions.PlotReport(['main/loss', 'validation/main/loss'],
    #                               'epoch', file_name='loss.png'))
    #     trainer.extend(
    #         extensions.PlotReport(
    #             ['main/accuracy', 'validation/main/accuracy'],
    #             'epoch', file_name='accuracy.png'))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    # trainer.extend(extensions.PrintReport(
    #     ['epoch', 'main/loss', 'validation/main/loss',
    #      'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    # trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

def my_concat(batch, device=None, padding=None):
    """
    将数据和标签分离
    :param batch:
    :param device:
    :param padding:
    :return:
    """
    # print(batch)
    data = []
    label = []
    for it in batch:
        data.append(it[0])
        label.append(it[1])
    data = np.array(data)
    label = np.array(label)
    return data,label
    # exit(0)
    # pass

def manula_train_loop():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=10,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--layer', '-l', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    n_rhythm = 1000
    model = Classifier(XSNet(args.layer, 54, n_rhythm, args.unit))
    if args.gpu >= 0:
        # Make a specified GPU currentsoftmax_cross_entropy
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load the MNIST dataset
    train, test = datasets.get_data()
    print('train', len(train))
    # exit(0)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)
    max_epoch = 10
    while train_iter.epoch < max_epoch:
        train_batch = train_iter.next()
        # print(train_batch)
        # data_train, target_train = convert(train_batch, args.gpu)
        data = convert(train_batch, args.gpu)
        # prediction_train = model(data_train,target_train)
        prediction_train = model(data['xs'],data['ys'])
        # print(prediction_train)
        loss = F.softmax_cross_entropy(prediction_train, target_train)
        model.cleargrads()

        loss.backward()
        optimizer.update()

        if train_iter.is_new_epoch:
            print('epoch:{:02d} train_loss:{:.04f} '.format(
                train_iter.epoch, float(to_cpu(loss.data))
            ), end='')
            test_losses = []
            test_accuracies = []
            while True:
                test_batch = test_iter.next()
                data_test, target_test = convert(test_batch, args.gpu)
                prediction_test = model(data_test)
                loss_test = F.softmax_cross_entropy(prediction_test, target_test)
                test_losses.append(to_cpu(loss_test.data))
                accuracy = F.accuracy(prediction_test, target_test)

                if test_iter.is_new_epoch:
                    test_iter.epoch = 0
                    test_iter.current_position = 0
                    test_iter.is_new_epoch = False
                    test_iter._pushed_position = None
                    break

            print('val_loss:{:.04f} val_accuracy:{:.04f}'.format(
                np.mean(test_losses), np.mean(test_accuracies)
            ))


if __name__ == '__main__':
    main()
    # manula_train_loop()
