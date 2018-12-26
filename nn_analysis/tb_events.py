#
# Tufts University | Department of Physics
# High Energy Physics Research Group
# Liquid Argon Computer Vision (LArCV)
#
# filename: tb_events.py
#
# purpose:  generates csv file from tree of events.out*.localmachine file
#
# note:     this script expects a folder structure like:
#            - accuracy:
#              ¬ class tag 1
#              ¬ class tag 2
#              ¬ class tag 3
###############################
# Import Script
###############################
import os
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def tabulate_events(dpath):
    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload()\
                         for dname in os.listdir(dpath)]

    tags = summary_iterators[0].Tags()['scalars']

    for it in summary_iterators:
        assert it.Tags()['scalars'] == tags

    out   = defaultdict(list)
    steps = []

    for tag in tags:
        steps = [e.step for e in summary_iterators[0].Scalars(tag)]

        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            assert len(set(e.step for e in events)) == 1

            out[tag].append([e.value for e in events])

    return out, steps

def to_csv(dpath):
    dirs         = os.listdir(dpath)
    d, steps     = tabulate_events(dpath)
    tags, values = zip(*d.items())
    np_values    = np.array(values)

    for index, tag in enumerate(tags):
        df = pd.DataFrame(np_values[index], index=steps, columns=dirs)
        df.to_csv(get_file_path(dpath, tag))
    return tag

def get_file_path(dpath, tag):
    file_name   = tag.replace("/", "_") + '.csv'
    folder_path = os.path.join(dpath, 'csv')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return os.path.join(folder_path, file_name)

def plot_events(path, tag):
    outFile = path + 'csv/' + tag
    x = np.loadtxt(open(outFile, 'r'), delimiter = ",", skiprows = (1), usecols = (1, 5))
    fig   = plt.figure()
    ax    = fig.add_subplot(111)
    title = 'Data Plot'
    ax.set_title(title)
    ax.set_xlabel('Iteration Number')
    ax.set_ylabel('Loss / Accuracy')
    plt.plot(x = x[:0], y = [x[:1], x[:2], x[:3], x[:4], x[:5]])
    saveFig = path + '_' + title + '.png'
    plt.savefig(saveFig, dpi=300)

###############################
# Main Function
###############################
def main():
    path = input("Provide file path or drag in folder from GUI:")
    path = path.rstrip() + '/'
    tag = to_csv(path)
    tag = tag.replace("/", "_") + '.csv'
    plot_events(path, tag)

if __name__ == '__main__':
    main()
