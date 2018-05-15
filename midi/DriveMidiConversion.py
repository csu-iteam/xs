import sys


sys.path.append('./midi')
import os
import MidiFileAnalysis as mda, DataBaseInit as dbi

MIDI_SRC_PATH = "./midiSrc"
MIDI_TXT_PATH = "./midiTxt"
MIDI_TO_TXT = "./mf2t "
TXT_TO_MIDI = "./t2mf "


# preproccess midi file
def init_midi():
    if os.path.exists(MIDI_SRC_PATH):
        message = os.walk(MIDI_SRC_PATH)
        mc = open("midi.config", "w")
        label = 0
        for x in message:
            temp = x[2]
            for y in temp:
                sl = y.split(".")
                name = sl[0]
                cur_dir = os.path.split(os.path.realpath(__file__))[0]
                cmd_path = os.path.join(cur_dir,MIDI_TO_TXT)
                src_path = os.path.join(os.path.join(cur_dir,MIDI_SRC_PATH),y)
                tar_path = os.path.join(os.path.join(cur_dir,MIDI_TXT_PATH),name+'.txt')
                cmd = cur_dir+'/'+MIDI_TO_TXT + MIDI_SRC_PATH + "/" + y + " > " + MIDI_TXT_PATH + "/" + name + ".txt"
                cmd = cmd_path + ' ' + src_path + ' > '+tar_path
                # print('cmd',cmd)
                os.system(cmd)
                # commands.getstatusoutput(
                #     MIDI_TO_TXT + MIDI_SRC_PATH + "/" + y + ">" + MIDI_TXT_PATH + "/" + name + ".txt")
                mc.write(str(label) + " " + name + ".txt" + "\n")
                label += 1
    else:
        print("midi src path does not exist!")


def extract(midi_label, frame_num):
    dbi.load_database("database.txt")
    file_list = []
    # Get Current Directory
    cur_dir = os.path.split(os.path.realpath(__file__))[0]
    fl = open(os.path.join(cur_dir, "midi.config"))
    for line in fl:
        temp = line.split()
        file_list.append(temp[1])
    midi_file = file_list[midi_label]
    l_list = mda.analysis("midiTxt/" + midi_file)
    l_list = mda.channel_merge(l_list)
    label_list = dbi.make_data(l_list, frame_num)
    dbi.export_database("database.txt")
    return label_list


# def make_midi():


if __name__ == '__main__':
    init_midi()
    # for x in range(0):
    # extract(0, 1500)
    exit(0)
