import numpy
import os
MFile = "MFile"
MTrk = "MTrk"
TrkEnd = "TrkEnd"
Tempo = "Tempo"
KeySig = "KeySig"
major = "major"
Meta = "Meta"
SeqName = "SeqName"
TimeSig = "TimeSig"
Par = "Par"
On = "On"
Off = "Off"
PrCh = "PrCh"


# return the value of n ,when receiving an input like str "n=x"
def find_value(temp_x):
    # print temp_x
    temp_x = temp_x.split('=')
    # print temp_x[0] + temp_x[1]
    return int(temp_x[1])


# translate pitch position into time duration in 1/600 second according to the tempo and distinguishability.
# using 1/600 second will be convenience for matching fps
def exchange_time(position, dis, tempo):
    speed = 60000000 / tempo
    time = 1.0 * position / dis * speed * 10
    return int(round(time))


# divide vol into ten levels
def divide_vol(vol):
    divided_vol = round(vol / 10.0)
    if divided_vol >= 10.0:
        return 9.0
    return divided_vol


def channel_merge(l_list):
    l_list = sorted(l_list, key=lambda unit: unit[1])
    return l_list


# analysis the midi txt,and return a list of pitch,start_time,end_time,start_vol,end_vol
def analysis(midi_txt_name):
    # Get Current Directory
    cur_dir = os.path.split(os.path.realpath(__file__))[0]
    midi_file = open(os.path.join(cur_dir,midi_txt_name))
    # lList is used for saving when the pitch is on and duration and vol start and vol end
    # l_list = [[0 for i in range(5)] for j in range(10000)]
    l_list = []
    l_point = 0
    # nList is used for saving which pitch has not end;
    n_list = [0] * 100

    for line in midi_file:
        temp = line.split()

        if temp[0] == MFile:
            dis = int(temp[3])

        elif temp[0] == MTrk:
            continue

        elif temp[0] == TrkEnd:
            continue

        elif temp[0] == "0":
            if temp[1] == Tempo:
                # do tempo
                tempo = int(temp[2])
            elif temp[1] == KeySig:
                # do KeySig
                continue
            elif temp[1] == Meta:
                if temp[2] == SeqName:
                    # do SeqName
                    continue
                elif temp[2] == TrkEnd:
                    # do TrkEnd
                    continue
            elif temp[1] == Par:
                continue
            elif temp[1] == PrCh:
                continue
            elif temp[1] == On:
                unit = [0 for i in range(5)]
                n = find_value(temp[3])
                unit[0] = n
                unit[1] = exchange_time(int(temp[0]), dis, tempo)
                n_list[n] = int(temp[0])
                v = find_value(temp[4])
                unit[3] = divide_vol(v)
                l_list.append(unit)
                l_point += 1

        else:
            if temp[1] == Meta:
                if temp[2] == TrkEnd:
                    continue
                else:
                    continue
            elif temp[1] == On:
                unit = [0 for i in range(5)]
                n = find_value(temp[3])
                unit[0] = n
                unit[1] = exchange_time(int(temp[0]), dis, tempo)
                n_list[n] = l_point
                v = find_value(temp[4])
                unit[3] = divide_vol(v)
                l_list.append(unit)
                l_point += 1
            elif temp[1] == Off:
                n = find_value(temp[3])
                end_time = int(temp[0])
                v = find_value(temp[4])
                li = n_list[n]
                l_list[li][2] = exchange_time(int(end_time), dis, tempo)
                l_list[li][4] = divide_vol(v)
            elif temp[1] == Par:
                continue
            else:
                print("row[2] error" + temp[1])
    midi_file.close()
    return l_list


def test_analysis():
    l_list = analysis("panama.txt")
    file_out = open("panama-result.txt", 'wt')
    l_list = channel_merge(l_list)
    for record in l_list:
        for item in record:
            file_out.write(str(item) + " ")
        file_out.write("\n")
    file_out.close()


# if __name__ == '__main__':
#     test_analysis()
#     exit(0)
