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
Build a server and export interface to user
"""
import os
import sys

sys.path.append('..')
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import hashlib
from model import XSNet
from train import load_midi_snippet
from midi.DriveMidiConversion import make_midi
# import../ convert_to_dataset_with_label  as mk_data
from extractor import FramesExtractor, PoseExtractor, DataExtractor
import mimi
import numpy as np
from pydub import AudioSegment
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
import datetime
app = Flask(__name__)

UPLOAD_FOLDER = '/root/data/video/'
ALLOWED_EXTENSIONS = set(['mp4'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['SECRET_KEY'] = os.urandom(24)
cur_dir = os.path.split(os.path.realpath(__file__))[0]


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def init_model():
    """
    加载标签数据库，加载模型
    :return:
    """
    path = os.path.join(cur_dir, '../midi/database.txt')
    target_midi_ids = load_midi_snippet(path)
    target_midi_ids = {i: w for w, i in target_midi_ids.items()}
    n_rhythm = len(target_midi_ids)
    model = XSNet(3, 54, n_rhythm, 1024)

    npz_path = cur_dir + '/result/model_epoch-294'
    print('load model: {}'.format(npz_path))
    serializers.load_npz(npz_path, model)
    return model, target_midi_ids


def call_t2mf(path, output_path):
    """
    调用t2mf，将txt文本转换为mid文件
    :param path:
    :param output_path:
    :return:
    """
    cmd = 't2mf {} {}'.format(path, output_path)
    print(cmd)
    os.system(cmd)


def call_midi2wav(path, output_path):
    """
    调用mimi,将mid文件转换为wav文件
    :param path:
    :param output_path:
    :return:
    """
    mimi.output.midi2wav(path, output_path)


def call_wav2mp3(path, output_path):
    """
    调用pydub，将wav文件转为mp3文件
    :param path:
    :param output_path:
    :return:
    """
    song = AudioSegment.from_wav(path)
    song.export(output_path, format="mp3")


def generate_music(path):
    """
    生成音乐，大致流程为
    将视频变成图片帧 12帧每秒
    使用openpose提取节点信息
    规整输入
    使用xsnet进行预测
    根据预测得到的标签，生成txt文本
    将txt文本转换为mid文件
    将mid文件转换为wav文件
    将wav文件转换为mp3文件
    返回mp3文件的URL地址，相对服务器的
    :param path:
    :return:
    """
    # If it is not a video, throw an exception file type error
    if not os.path.exists(path):
        raise Exception('file: {}  not exist'.format(path))
    if not path.endswith('.mp4'):
        raise Exception('file is invalid')
    # Get filename
    print('Encrypted file name...')
    filename = os.path.basename(path)
    h1 = hashlib.md5()
    h1.update(filename.encode(encoding='utf-8'))
    e_filename = h1.hexdigest()
    print('cur name:{}'.format(e_filename))
    
    print('Extract frames...')
    begin = datetime.datetime.now() 
    frames_output_path = '/root/data/flask/frames/' + e_filename
    ex = FramesExtractor()
    ex.extract(path, frames_output_path)
    end = datetime.datetime.now()
    print('extract frames cost time: {}'.format(end-begin))
    
    begin = end
    print('Extract pose...')
    pose_output_path = '/root/data/flask/json/' + e_filename
    ex = PoseExtractor('/root/data/openpose')
    ex.extract(frames_output_path, pose_output_path)
    end = datetime.datetime.now()
    print('extract pose cost time: {}'.format(end-begin))

    print('Extract data...')
    begin = end
    model, target_midi_ids = init_model()
    # data = make_data(pose_output_path)
    ex = DataExtractor()
    # 这里需要测试
    data = ex.extract(pose_output_path)
    print('data: {} '.format(data))
    print('extract data cost time: {}'.format(end-begin))

    begin = end
    ret = model.translate(data)
    print('predict result: {}'.format(ret[0]))
    end = datetime.datetime.now()
    print('predict seq cost time: {}'.format(end-begin))

    os.chdir(cur_dir)
    begin = end
    midi_txt_path = '/root/data/flask/txt/' + e_filename
    make_midi(midi_txt_path, ret[0])
    # 调用t2mf
    midi_path = '/root/data/flask/midi/' + e_filename + '.mid'
    call_t2mf(midi_txt_path + '.txt', midi_path)
    wav_path = '/root/data/flask/wav/' + e_filename + '.wav'
    call_midi2wav(midi_path, wav_path)
    mp3_path = '/root/data/xs/xsnet/static/' + e_filename + '.mp3'
    call_wav2mp3(wav_path, mp3_path)
    midi_output_path = 'http://47.95.203.153/static/{}'.format(e_filename+'.mp3')
    return midi_output_path


@app.route('/upload_file', methods=['GET'])
def upload_index():
    return """
        <!doctype html>
        <title>Upload new File</title>
        <h1>Upload new File</h1>
        <form method=post enctype=multipart/form-data action=/upload_file>
          <input type=file name=file>
          <input type=submit value=Upload>
        </form>
        """


@app.route('/upload_file', methods=['POST'])
def upload_file():
    """
    Accept the video file uploaded by the client and return the score result
    :return:
    """
    ret = {'code': 200, 'status': True, 'data': [], 'msg': 'ok'}
    if 'file' not in request.files:
        flash('No file part')
        ret['status'] = False
        ret['code'] = 400
        ret['msg'] = 'No file part'
        return jsonify(ret)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        ret['status'] = False
        ret['code'] = 400
        ret['msg'] = 'No file part'
        return jsonify(ret)
    if file and allowed_file(file.filename):
        begin = datetime.datetime.now()
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        url = generate_music(path)
        # url = 'http://47.95.203.153/static/my_test.mp3'
        print('url: {}'.format(url))
        ret['data'] = [url, url, url]
        end = datetime.datetime.now()
        print('total cost time: {}'.format(end-begin))
        return jsonify(ret)
    else:
        ret['status'] = False
        ret['code'] = 400
        ret['msg'] = 'File not valid. Please upload .mp4 file.'
        return jsonify(ret)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
    # path = '/root/data/video/v1.mp4'
    # generate_music(path)
