# -*- coding: utf-8 -*-
# 将已经得到的pose信息转化为numpy的npz形式
# 转化时需要按照顺序，信息文件格式[视频名].[序号].keypoints.json
# 比如：Video0.mp4.0001_keypoints.json
# 其中，这个序号是4位数，按照0000开始增长
# pose信息在json中的位置:people.[pose_keypoints_2d]
# 可能存在多人的情况，处理方式为,只取第一个人的信息
# 如果没有检测到pose信息，则以上一次的pose信息作为本次的信息
import os, sys
import json
import numpy as np
import datetime
from midi import DriveMidiConversion
POSE_ROOT = '/root/data/google_driver/json/'
starttime = datetime.datetime.now()
if not os.path.exists(POSE_ROOT):
    raise Exception('path:' + POSE_ROOT + " not exist")


def get_pose_info(path):
    # print("handle: "+path)
    if not os.path.exists(path):
        raise Exception("file: " + path + " not exist")
    f = open(path, 'r')
    content = f.read()
    f.close()
    data = json.loads(content)
    # print('data: '+str(data))
    people = data['people']
    if people is None:
        raise Exception("data:" + data + " has not people info")
    if len(people) == 0:
        print("people has not pose info")
        return None
    else:
        return people[0]["pose_keypoints_2d"]

def get_type_num(dir):
    data={
        'BboomBboom':0,
        'Confession_Balloon':2,
        'seve':5,
        'goodtime':6,
        'jilejingtu':5,
        'panama':1,
        'shapeofyou':3
    }
    return data[dir]

pose_infos = []
for dir1 in os.listdir(POSE_ROOT):
    # 如果不是目录，则忽略
    if os.path.isfile(POSE_ROOT + '/' + dir1):
        print("ignore file:" + dir1)
        continue
    # 这里是视频类的级别
    type_level_infos = []
    for dir2 in os.listdir(POSE_ROOT + '/' + dir1):
        # e.g: Video0.mp4.0001_keypoints.json
        dir = POSE_ROOT + '/' + dir1 + '/' + dir2
        # 忽略文件
        if os.path.isfile(dir):
            print("ignore file:" + dir2)
            continue
        # 构造序号
        n_files = len(os.listdir(dir))
        # 每个文件夹下的pose信息用单独的数组存
        # 这里是视频级别
        infos = []
        last_info = None
        for index in range(n_files):
            # 构造文件名
            filename = dir2 + '.' + str(index + 1).zfill(4) + "_keypoints.json"
            path = dir + '/' + filename
            info = get_pose_info(path)
            if info is None:
                if last_info is None:
                    # raise Exception("Pose Info Error")
                    info = [0 for x in range(54)]
                info = last_info
            last_info = info
            # 将节点信息存储下来
            infos.append(info)
        # 这里添加标签信息
        num = get_type_num(dir1)
        if num is None:
            raise Exception('Unsuport type ',dir1)
        labels = DriveMidiConversion.extract(num,len(infos))
        type_level_infos.append((infos,labels))
    pose_infos.append(type_level_infos)
# 最后转换为npz
data = np.array(pose_infos)
np.savez('pose.npz', data)
endtime = datetime.datetime.now()
print('times: ', (endtime - starttime).seconds, ' seconds')
