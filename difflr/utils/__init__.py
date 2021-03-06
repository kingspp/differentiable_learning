import traceback
import sys
import json
import uuid
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from prettytable import PrettyTable


def split_array(array, splits):
    tot = 0
    splitted_array = []
    for e, s in enumerate(splits):
        if e == 0:
            splitted_array.append(array[0:s])
        else:
            splitted_array.append(array[tot:tot + s])
        tot += s
    return splitted_array


def plot_information_transfer(model, weights, display=True, debug=False):
    node_table = PrettyTable()
    weight_table = PrettyTable()
    node_table.field_names = ['Contents', 'L0'] + [f"L_{e + 1} " for e, i in enumerate(weights[:])]
    weight_table.field_names = ['Contents', 'L0'] + [f"L_{e + 1} " for e, i in enumerate(weights[:])]
    rows = ['L0', 'L1', 'L2', 'L3']
    cols = ['L0', 'L1', 'L2', 'L3']

    splits = {}
    for i, w in enumerate(weights[:]):
        split = [model.layers[0].in_features]
        for j, layer in enumerate(model.layers[:i]):
            split.append(layer.out_features)
        splitted_weights = split_array(w, split)
        splits = {**splits,
                  **{f'L{e}-to-L{i + 1}': {'nodes': s, 'weights': splitted_weights[e]} for e, s in enumerate(split)}}

    splits = dict(sorted(splits.items()))

    nodes = {c: {} for c in cols}
    _weights = {c: {} for c in cols}
    for r in rows:
        for c in cols:
            s = f"{c}-to-{r}"
            if r == c or s not in splits:
                nodes[c][r] = '--'
                _weights[c][r] = '--'
            else:
                nodes[c][r] = splits[s]['nodes']
                _weights[c][r] = round(splits[s]['weights'].mean(), 2)

    node_table.add_row([f"L{0}", *list(nodes[f"L{0}"].values())])
    weight_table.add_row([f"L{0}", *list(_weights[f"L{0}"].values())])
    for e, w in enumerate(weights[:]):
        node_table.add_row([f"L{e + 1}", *list(nodes[f"L{e + 1}"].values())])
        weight_table.add_row([f"L{e + 1}", *list(_weights[f"L{e + 1}"].values())])

    overall_transfer = np.mean([np.mean(w) for w in weights])
    if debug:
        print("\nNodes:")
        print(node_table)

    if display:
        print(f'\nInformation Transfer: {overall_transfer:.4f}')
        print(weight_table)

    return weight_table, overall_transfer


def mse_score(logits, target, num_classes, reduction=None, device=torch.device('cpu')):
    one_hot_targets = np.eye(num_classes)[target.cpu()]
    one_hot_targets = torch.tensor(one_hot_targets, dtype=torch.float).reshape([-1, num_classes]).to(device)
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
            return "Long Seq . . ."  # obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        else:
            try:
                # return  obj.default()
                return "Long Seq"
            except Exception:
                return f'Object not serializable - {obj}'


def confidence_score(predictions, labels):
    correct_indices = {l: [] for l in np.unique(labels)}

    for e, (p, l) in enumerate(zip(predictions, labels)):
        if np.argmax(p) == l:
            correct_indices[l].append(e)

    scores = {}
    for k, v in correct_indices.items():
        if len(v) > 0:
            scores[str(k)] = round(float(np.max(predictions[v], 1).mean().flatten()), 2)
    return scores, round(float(np.mean(list(scores.values()))), 2)


def check_gpu():
    if not torch.cuda.is_available():
        print('WARN!! GPU not found!')
        exit()
