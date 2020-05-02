import traceback
import sys
import json
import uuid
import datetime


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
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self

    def __exit__(self, exc_type, exc_value, tb):
        sys.stdout = self.stdout
        if exc_type is not None:
            self.file.write(traceback.format_exc())
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
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
