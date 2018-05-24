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
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import hashlib
from model import XSNet
from train import load_midi_snippet

# import../ convert_to_dataset_with_label  as mk_data
from extractor import FramesExtractor, PoseExtractor, DataExtractor

import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = '/root/data/video/'
ALLOWED_EXTENSIONS = set(['mp4'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['SECRET_KEY'] = os.urandom(24)
cur_dir = os.path.split(os.path.realpath(__file__))[0]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def init_model():
    path = os.path.join(cur_dir, '../midi/database.txt')
    target_midi_ids = load_midi_snippet(path)
    target_midi_ids = {i: w for w, i in target_midi_ids.items()}
    n_rhythm = len(target_midi_ids)
    model = XSNet(3, 54, n_rhythm, 1024)
    return model, target_midi_ids


def make_data(path):
    if not os.path.exists(path):
        raise Exception(path, ' is not exist')
    infos = []
    for file in os.listdir(path):
        info = mk_data.get_pose_info(file)

        if info is None:
            if last_info is None:
                # raise Exception("Pose Info Error")
                info = [0 for x in range(54)]
            info = last_info
        last_info = info
        # 将节点信息存储下来
        infos.append(info)
    return np.array(infos)


def generate_music(path):
    # If it is not a video, throw an exception file type error
    if not os.path.exists(path):
        raise Exception('file: {}  not exist'.format(path))
    if not path.endswith('.mp4'):
        raise Exception('file is invalid')
    # Get filename
    filename = os.path.basename(path)
    h1 = hashlib.md5()
    h1.update(filename.encode(encoding='utf-8'))
    e_filename = h1.hexdigest()
    frames_output_path = '/root/data/flask/frames/' + e_filename
    ex = FramesExtractor()
    ex.extract(path, frames_output_path)
    # extract_frame(path, frames_output_path)
    pose_output_path = '/root/data/flask/json/' + e_filename
    ex = PoseExtractor('/root/data/openpose')
    ex.extract(frames_output_path, pose_output_path)
    # extract_pose(frames_output_path, pose_output_path)
    model, target_midi_ids = init_model()
    # data = make_data(pose_output_path)
    ex = DataExtractor()
    # 这里需要测试
    data = ex.extract(pose_output_path)
    print('data: {} '.format(data))
    ret = model.translate(data)
    
    ret = map(lambda x: x.argmax(), ret.data)
    print('ret: {}'.format(ret))
    # 将序列转换为mp3文件，然后放在当前目录下的static文件夹中，返回给客户端
    midi = []
    for it in ret:
        midi.append(target_midi_ids[it])
    path = cur_dir + '/' + e_filename
    # midi_output_path = '/root/data/flask/midi/' + e_filename
    midi_output_path = path
    np.savez(midi_output_path, np.array(midi))
    return midi_output_path


def extract_frame(video_path, output_path):
    basename = os.path.basename(video_path)
    frames = 12
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cmd = 'ffmpeg -i ' + video_path + ' -r ' + str(frames) + ' ' + output_path + '/' + basename + '.%4d.jpg > /dev/null'
    os.system(cmd)


def extract_pose(frame_dir, output_path):
    OPENPOSE_ROOT = '/root/data/openpose/'
    #  ./build/examples/openpose/openpose.bin --image_dir /home/pikachu/Desktop/test --write_json /home/pikachu/Desktop/test --net_resolution 192x144 --display 0
    bin_path = OPENPOSE_ROOT + '/build/examples/openpose/openpose.bin'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cmd = bin_path + ' --image_dir ' + frame_dir + ' --write_json ' + output_path + ' --display 0 --keypoint_scale 3 > /dev/null'
    os.system(cmd)
    pass


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
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        # path = generate_music(path)
        url = 'http://47.95.203.153/static/panama.mp3'
        ret['data'] = [url, url, url]
        return jsonify(ret)
    else:
        ret['status'] = False
        ret['code'] = 400
        ret['msg'] = 'File not valid. Please upload .mp4 file.'
        return jsonify(ret)

if __name__=='__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
    # path = '/root/data/video/v1.mp4'
    # generate_music(path)
