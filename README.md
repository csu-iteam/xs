# 形声APP
项目目录结构
```
xs
├── client
├── midi
└── xsnet
```
说明
1. xs为项目主目录，该目录下存放client,xsnet,midi共3个文件夹和其他常用的数据处理脚本
2. client为安卓源码目录
3. xsnet为自己编写的深度学习部分代码

项目主目录下文件用途说明

| 文件名                              | 用途               |
| -------------------------------- | ---------------- |
| convert_to_dataset.py            | 将节点信息转换为数据集，不带标签 |
| convert_to_dataset_with_label.py | 将节点信息转换为数据集，带标签  |
| extract_all.py                   | 从视频中提取帧          |
| extract_pose.py                  | 从帧中提取节点信息        |

xsnet目录下主要文件用途说明

| 文件名         | 用途           |
| ----------- | ------------ |
| datasets.py | 为深度学习训练提供数据集 |
| model.py    | 为深度学习提供模型    |
| train.py    | 为深度学习提供训练代码  |
| server.py   | 对外提供web服务的代码 |

midi目录下主要文件用途说明

| 文件名                   | 用途                                       |
| --------------------- | ---------------------------------------- |
| MidiFileAnalysis.py   | 将midi文本文件解析成二维数组，每行代表一个记录，包括音调，发声位置，关闭位置，发声音量，关闭音量。使用音符组合序列生成midi文本文件。 |
| DataBaseInit.py       | 将包含midi信息的二维数组中的音符组合提取到database.txt中，并标记，把midi二维数组转化为标签序列。依据database.txt将音符组合标签序列转化为音符组合序列。 |
| DriveMidConversion.py | 提供从midi文件中提取音符组合标签序列的方法，提供将音符组合标签序列反向生成midi文本文件的接口。提供将midi音乐文件转化为midi文本文件的接口。 |
| midi.config           | 记录midi音乐的标识                              |
| database.txt          | 记录音符组合的标签                                |

补充说明，在该目录下还有两个子目录，它们的作用分别是：　


| 文件夹     | 用途           |
| ------- | ------------ |
| midiSrc | 用于存储midi音乐文件 |
| midiTxt | 用于存储midi文本文件 |

