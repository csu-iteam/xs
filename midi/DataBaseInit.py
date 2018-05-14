import os

fps = 12  # frame per second
data = []  # the database of pitch combination
live_pitch = []  # the pitch which is on


# p = 0  # the pointer of l_list


def add_live_pitch(p, now, l_list):
    for x in range(p, len(l_list)):
        if l_list[x][1] <= now:
            live_pitch.append([l_list[x][0], l_list[x][2]])
            p += 1
        else:
            break
    return p


def remove_dead_pitch(now):
    for x in live_pitch:
        if x[1] <= now:
            live_pitch.remove(x)
            remove_dead_pitch(now)
            break


def pitch_merge():
    unit = []
    for x in live_pitch:
        flag = False
        for y in unit:
            if x == y:
                flag = True
        if not flag:
            unit.append(x[0])
    unit.sort()
    return unit


# search if the pitch combination already exists in the database
def match_pitch_combination(pc):
    for x in range(0, len(data)):
        if len(data[x]) == len(pc):
            flag = True
            for y in range(0, len(pc)):
                if data[x][y] != pc[y]:
                    flag = False
                    break
            if flag:
                return x
        else:
            continue
    return -1


# main entry
def make_data(l_list, frame):
    point_start = 0
    now = 0
    step = 600 / fps
    sign_list = []  # the list of id appeared
    for i in range(0, len(live_pitch)):  # init live_pitch
        live_pitch.pop()
    for i in range(frame):
        point_start = add_live_pitch(point_start, now, l_list)
        remove_dead_pitch(now)
        pc = pitch_merge()
        pid = match_pitch_combination(pc)
        if pid == -1:
            data.append(pc)
            pid = len(data) - 1
        sign_list.append(pid)
        now += step
    return sign_list


def export_database(txt_name):
    fi = open(txt_name, "w")
    for x in data:
        for y in x:
            fi.write(str(y) + " ")
        fi.write("\n")
    fi.close()


def load_database(txt_name):
    for x in range(0, len(data)):
        data.pop()
    if not os.path.exists(txt_name):
        fi = open(txt_name, 'w')
        fi.close()
    else:
        fi = open(txt_name)
        for line in fi:
            temp = line.split()
            unit = []
            for x in temp:
                unit.append(int(x))
            data.append(unit)
        fi.close()
