import os, sys
sys.path.append('..')
import hashlib
from chainer serializers
from extractor import FramesExtractor, PoseExtractor, DataExtractor
from generator import MidiGenerator, WavGenerator, Mp3Generator
from datasets import load_midi_snippet
from model import XSNet
from midi.DriveMidiConversion import make_midi

cur_dir = os.path.split(os.path.realpath(__file__))[0]
class XSNetPredictor(object):
    def __init__(self, openpose_root, midi_database_path, model_path):
        self.openpose_root = openpose_root
        self.midi_database_path = midi_database_path
        self.model_path = model_path

    def predict(self, mp4_path, ouput_mp3_path, tmp_path=None):
        if not os.path.exists(mp4_path):
            raise Exception('file: {} not exist'.format(mp4_path))
        dir_name = os.path.dirname(ouput_mp3_path)
        basename = os.path.basename(ouput_mp3_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        h = hashlib.md5()
        h.update(basename.encode(encoding='utf-8'))
        e_name = h.hexdigest()

        if tmp_path is None:
            tmp_path = os.path.join(dir_name, e_name)
        else:
            tmp_path = os.path.join(tmp_path, e_name)
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)

        frames_path = os.path.join(tmp_path, 'frames')
        pose_path = os.path.join(tmp_path, 'json')
        FramesExtractor().extract(mp4_path, frames_path)
        PoseExtractor(self.openpose_root).extract(frames_path, pose_path)
        data = DataExtractor().extract(pose_path)

        model = self._init_model()
        ret = model(data)

        os.chdir(cur_dir)
        midi_txt_path = os.path.join(tmp_path,e_name)
        midi_path = os.path.join(tmp_path,e_name+'.mid')
        wav_path = os.path.join(tmp_path,e_name+'.wav')
        make_midi(midi_txt_path, ret[0])
        MidiGenerator().generate(midi_txt_path+'.txt',midi_path)
        WavGenerator().generate(midi_path, wav_path)
        Mp3Generator().generate(wav_path,ouput_mp3_path)

    def _init_model(self):
        midi_ids = load_midi_snippet(self.midi_database_path)
        n_rhythm = len(midi_ids)
        model = XSNet(3, 54, n_rhythm, 1024)
        serializers.load_npz(self.model_path, model)
        return model

if __name__ == '__main__':
    openpose_root = '/root/data/openpose'
    midi_database_path = '../midi/database.txt'
    model_path = 'result/model_epoch-294'
    p = XSNetPredictor(openpose_root,midi_database_path,model_path)
    p.predict('/root/data/Video113.mp4','/root/data/test.mp3')