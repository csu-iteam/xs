# -*- coding: utf-8 -*-
import os, sys
import datetime
# 所有数据放在DATASET_ROOT下，每一类放在一个文件夹
# 提取的帧放在OUTPUT_ROOT/pose下，每一类在一个文件夹，每一个视频提取的帧在以视频命令的文件夹下
DATASET_ROOT='/root/data/google_driver/frames/'
OUTPUT_ROOT='/root/data/google_driver/'
OPENPOSE_ROOT='/root/data/openpose/'
# 提取帧的命令
def get_cmd(dir):
	#  ./build/examples/openpose/openpose.bin --image_dir /home/pikachu/Desktop/test --write_json /home/pikachu/Desktop/test --net_resolution 192x144 --display 0
	bin_path = OPENPOSE_ROOT+'/build/examples/openpose/openpose.bin'
	image_dir = DATASET_ROOT + dir
	write_json_path = OUTPUT_ROOT+'json/'+dir
	if not os.path.exists(write_json_path):
		os.makedirs(write_json_path)
	cmd=bin_path+' --image_dir '+image_dir+' --write_json '+write_json_path+' --display 0 --keypoint_scale 3 > /dev/null'
	return cmd
starttime = datetime.datetime.now()
os.chdir(OPENPOSE_ROOT)
# 遍历DATASET_ROOT
# 实际上，根据我的目录结构，其实只需要跟进两层
# 一层是大类，一层是具体的
dirs = os.listdir(DATASET_ROOT)
for dir1 in dirs:
	for dir2 in os.listdir(DATASET_ROOT+'/'+dir1):
		# 视频帧所在目录
		dir = dir1+'/'+dir2
		cmd = get_cmd(dir)
		print(cmd)
		os.system(cmd)


endtime = datetime.datetime.now()
print('times: ',(endtime-starttime).seconds,' seconds')
