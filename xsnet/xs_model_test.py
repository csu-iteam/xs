from chainer import datasets, iterators, optimizers, serializers, config
from model import XSNet, Classifier
from train import handle_data, load_midi_snippet, convert
import datasets


def my_test():
    train, test = datasets.get_data()
    target_midi_ids = load_midi_snippet('../midi/database.txt')
    target_midi_ids = {i: w for w, i in target_midi_ids.items()}
    train = handle_data(train)
    test = handle_data(test)
    n_rhythm = len(target_midi_ids)
    # model = Classifier(XSNet(args.layer, 54, n_rhythm, args.unit))
    model = XSNet(3, 54, n_rhythm, 1024)
    config.train = False
    # model = XSNet()
    # serializers.load_npz('/home/pikachu/Documents/snapshot_iter_6250.jilejingtu', model)
    # serializers.load_npz('/home/pikachu/Documents/snapshot_iter_2719', model)
    test = convert(test,-1)
    test_data = test['xs']
    # print(test_data)
    ret = model.translate(test_data)
    print(ret)


if __name__ == '__main__':
    my_test()
