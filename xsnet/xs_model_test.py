from chainer import datasets, iterators, optimizers, serializers, config
from model import XSNet, Classifier
from train import handle_data, load_midi_snippet, convert
import datasets
from extractor import DataExtractor


def my_test():
    train, test = datasets.get_data()
    target_midi_ids = load_midi_snippet('../midi/database.txt')
    target_midi_ids = {i: w for w, i in target_midi_ids.items()}
    train = handle_data(train)
    test = handle_data(test)
    n_rhythm = len(target_midi_ids)
    # model = Classifier(XSNet(args.layer, 54, n_rhythm, args.unit))
    # model = XSNet(3, 54, n_rhythm, 1024)
    model = XSNet(3, 54, n_rhythm, 10)
    config.train = False
    # model = XSNet()
    # serializers.load_npz('/home/pikachu/Documents/snapshot_iter_6250.jilejingtu', model)
    # serializers.load_npz('/home/pikachu/Documents/snapshot_iter_2719', model)
    test = convert(test, -1)
    test_data = test['xs'][0]
    # print(test_data)
    ret = model.translate(test_data)
    # ret = model(test_data)
    # print(ret)
    ret = map(lambda x: x.argmax(), ret.data)
    # print(ret)
    print(list(ret))


def my_test_2():
    ex = DataExtractor()
    # ret = ex.extract('/home/pikachu/Documents/json/seve/Video1_clip.mp4')
    ret = ex.extract('/root/data/google_driver/json/seve/Video149.mpg')
    target_midi_ids = load_midi_snippet('../midi/database.txt')
    target_midi_ids = {i: w for w, i in target_midi_ids.items()}
    n_rhythm = len(target_midi_ids)
    print('rhythm: {}'.format(n_rhythm))
    model = XSNet(3, 54, n_rhythm, 1024)
    serializers.load_npz('result/snapshot_iter_27375.new', model)
    # serializers.load_npz('result/snapshot_iter_239', model)
    config.train = False
    ret = model.translate(ret)
    ret = map(lambda x: x.argmax(), ret.data)
    print(list(ret))


if __name__ == '__main__':
    my_test_2()
