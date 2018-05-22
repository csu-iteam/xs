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
Extract data in an object-oriented manner
"""
import os, sys
import os.path
import json
import numpy as np


class FramesExtractor(object):
    """
    Use ffmpeg to extract frames
    """

    def extract(self, video_path, output_path, frames=12, with_output=False, show_cmd=True):
        if not os.path.exists(video_path):
            raise Exception('video file not exist')
        basename = os.path.basename(video_path)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        cmd = 'ffmpeg -i {} -r {} {}/{}.%4d.jpg'.format(video_path, frames, output_path, basename)
        if not with_output:
            cmd = '{} > /dev/null'.format(cmd)

        if show_cmd:
            print(cmd)
        os.system(cmd)

    def extract_folder(self, folder_path, output_path):
        if not os.path.exists(folder_path):
            raise Exception('folder not exist')
        for it in os.listdir(folder_path):
            video_path = os.path.join(folder_path, it)
            frames_output_path = os.path.join(output_path, it)
            self.extract(video_path, frames_output_path)


class PoseExtractor(object):
    """
    Use openpose to extract pose info
    """

    def __init__(self, openpose_root_path):
        self.openpose_root_path = openpose_root_path
        self.openpose_bin_path = os.path.join(openpose_root_path, 'build/examples/openpose/openpose.bin')

    def extract(self, frames_path, output_path, with_output=False, show_cmd=True):
        os.chdir(self.openpose_root_path)
        cmd = '{} --image_dir {} --write_json {} --display 0 --keypoint_scale 3'.format(self.openpose_bin_path,
                                                                                        frames_path, output_path)
        if not with_output:
            cmd = '{} > /dev/null'.format(cmd)

        if show_cmd:
            print(cmd)

        os.system(cmd)

    def extract_foler(self, folder_path, output_path):
        if not os.path.exists(folder_path):
            raise Exception('folder not exist')
        for it in os.listdir(folder_path):
            frames_path = os.path.join(folder_path, it)
            json_output_path = os.path.join(output_path, it)
            self.extract(frames_path, json_output_path)


class DataExtractor(object):
    def _get_pose_info(self, json_file_path):
        if not os.path.exists(json_file_path):
            raise Exception("file: {} not exist".format(json_file_path))
        f = open(json_file_path, 'r')
        data = f.read()
        if len(data) == 0:
            raise Exception('file content is empty')
        f.close()
        data = json.loads(data)
        people = data['people']
        if people is None:
            raise Exception('data: {} has not people info'.format(data))
        if len(people) == 0:
            print('file: {} has not pose info'.format(json_file_path))
            return None
        else:
            return people[0]['pose_keypoints_2d']

    def extract(self, json_path):
        if not os.path.exists(json_path):
            raise Exception('path not exist')
        last = None
        data = []
        for it in os.listdir(json_path):
            file_path = os.path.join(json_path, it)
            ret = self._get_pose_info(file_path)
            if ret is None:
                if last is None:
                    ret = [0 for i in range(54)]
                else:
                    ret = last
            last = ret
            data.append(np.array(ret).astype(np.float32))

        return np.array(data).astype(np.float32)

    def extract_folder(self, ):

if __name__ == '__main__':
    ex = FramesExtractor()
    video_path = '/home/pikachu/Videos/形声.mp4'
    output_path = '/home/pikachu/Videos/frames/形声.mp4'
    # ex.extract(video_path, output_path)
    # ex = PoseExtractor()
    ex = DataExtractor()
    ret = ex.extract('/home/pikachu/Documents/json/seve/Video1_clip.mp4')
    print(ret.shape)
