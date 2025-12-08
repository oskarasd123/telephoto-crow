from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os



def get_run(dir):
    event_acc = EventAccumulator(dir, size_guidance={'tensors': 0})
    event_acc.Reload()
    values = [event.value for event in event_acc.Scalars("accuracy/data0")]
    return values



root_dir = "./logs"
def overview(root_dir, fig_name):
    log_dirs = os.listdir(root_dir)
    accuracies = []
    for dir in log_dirs:
        path = os.path.join(root_dir, dir)
        accuracy = get_run(path)
        accuracies.append(accuracy)
    print(np.mean([max(acc) for acc in accuracies]))
    fig = plt.figure()
    ax = plt.axes()
    ax.set_ylim(0.5, 1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("test acuracy")
    ax.set_title(fig_name)
    fig.add_axes(ax)
    for acc in accuracies:
        ax.plot(acc)
    if not os.path.exists("graphs"):
        os.mkdir("graphs")
    fig.savefig(os.path.join("graphs", f"{root_dir}.pdf"))


overview("logs", "not pretrained")
overview("logs_with_extra", "from pretrained")



