# -*- coding: utf-8 -*-
import os, sys
import datetime
# 所有视频按每秒14帧提取
# 所有数据放在DATASET_ROOT下，每一类放在一个文件夹
# 提取的帧放在OUTPUT_ROOT/frames下，每一类在一个文件夹，每一个视频提取的帧在以视频命令的文件夹下
DATASET_ROOT='/root/data/google_driver/video'
OUTPUT_ROOT='/root/data/google_driver/'
# 提取帧的命令
def get_cmd(file, frames = 12):
	# 获取最短的文件名
	basename = os.path.basename(file)
	output_path = OUTPUT_ROOT+'frames/'+file
	input_path = DATASET_ROOT+file
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	cmd='ffmpeg -i '+input_path+' -r '+str(frames)+' '+output_path+'/'+basename+'.%4d.jpg > /dev/null'
	return cmd
starttime = datetime.datetime.now()
# 遍历DATASET_ROOT
dirs = os.listdir(DATASET_ROOT)
for dir in dirs:
	for file in os.listdir(DATASET_ROOT+'/'+dir):
		# 如果不是.mp4和.mpg后缀，忽略
		if not (file.endswith('mp4') or file.encode('mpg')):
			print('ignore ',file)
		else:
			# 提取帧
			cmd = get_cmd(dir+'/'+file)
			print(cmd)
			os.system(cmd)

endtime = datetime.datetime.now()
print((endtime-starttime).seconds)
