import traceback
import sys
import json
import uuid
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from prettytable import PrettyTable

def plot_information_transfer(model, weights, display=True, debug=True):
    x = PrettyTable()
    x.field_names = ['Contents']+[f"L_{e+1} "for  e, i in enumerate(weights[:])]

    rows = ['L0','L1', 'L2', 'L3']
    cols = ['L0','L1', 'L2', 'L3']
    splits = {}
    for i, w in enumerate(weights[:]):
        split = [model.layers[0].in_features, model.layers[0].out_features]
        for j, layer in enumerate(model.layers[1:i+1]):
            split.append(layer.out_features)
        splits={**splits, **{f'L{i}_L{e}':s for e, s in enumerate(split)}}

    final_splits = {c: {}for c in cols}
    for r in rows:
        for c in cols:
            s = f"{r}_{c}"
            if r==c or s not in splits:
                final_splits[c][r]='--'
            else:
                final_splits[c][r] = splits[s]

    print(final_splits)
    print([w.shape for w in weights])
    exit()

    for e, w in enumerate(weights[:]):
        x.add_row([f"L{e+1}", *list(final_splits[f"L{e+1}"].values())])

    if debug:
        print("\nNodes:")
        print(x)

    if display:
        print('\nInformation Transfer:')
        print(x)

    return x


        

def mse_score(logits, target, num_classes, reduction=None):
    one_hot_targets = np.eye(num_classes)[target]
    one_hot_targets = torch.tensor(one_hot_targets, dtype=torch.float).reshape([-1, num_classes])
    return F.mse_loss(logits, one_hot_targets, reduction=reduction)

def generate_uuid(name: str = '') -> str:
    """
    | **@author:** Prathyush SP
    |
    | Generate Unique ID
    :param name: UID Name
    :return: Unique ID
    """
    return '_'.join([name, str(uuid.uuid4().hex)])


def generate_timestamp() -> str:
    """
    | **@author:** Prathyush SP
    |
    | Generate Timestamp
    :return: Timestamp in String : Format(YYYY-MM-DD_HH:mm:SEC)
    """
    return str(datetime.datetime.now().replace(microsecond=0)).replace(' ', '_').replace(':', '').replace('-', '')


# Context manager that copies stdout and any exceptions to a log file
class Tee(object):
    def __init__(self, filename, io_enabled=True):
        if io_enabled:
            self.file = open(filename, 'w')
        self.stdout = sys.stdout
        self.io_enabled = io_enabled

    def __enter__(self):
        sys.stdout = self

    def __exit__(self, exc_type, exc_value, tb):
        sys.stdout = self.stdout
        if self.io_enabled:
            if exc_type is not None:
                self.file.write(traceback.format_exc())
            self.file.close()

    def write(self, data):
        if self.io_enabled:
            self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        if self.io_enabled:
            self.file.flush()
        self.stdout.flush()


class CustomJsonEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        else:
            try:
                return obj.default()
            except Exception:
                return f'Object not serializable - {obj}'
