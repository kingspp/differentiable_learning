from copy import deepcopy
from sklearn.model_selection import KFold
import json
from difflr.utils import CustomJsonEncoder, generate_timestamp, Tee
from difflr import DIFFLR_EXPERIMENTS_RUNS_PATH
import torch
from itertools import product, count
from difflr.utils.early_stopping import EarlyStopping


class Tuner():
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.best_params = {}
        self.timestamp = generate_timestamp()
        self.global_counter = 0

    def tune(self, dataset, cv_split=3, epoch_end_hook=lambda x: x, data_per=100):
        with Tee(filename=DIFFLR_EXPERIMENTS_RUNS_PATH + f'/tuner_{self.timestamp}.log'):
            kf = KFold(n_splits=cv_split)
            data_indices = list(range(int(data_per * dataset.total_train_size / 100)))
            total_runs = len(
                    list(
                            product(range(cv_split), range(len(self.config['batch_size'])),
                                    range(len(self.config['lr'])))))
            for e, (train_index, valid_index) in enumerate(kf.split(data_indices)):
                print(f"Working on split {e}")
                for batch_size in self.config['batch_size']:
                    print(f"Working on batch size {batch_size}")
                    for lr in self.config['lr']:
                        self.global_counter += 1
                        print(f"Working on LR: {lr}")
                        current_config = deepcopy(self.config)
                        current_config['model_name'] = current_config['model_name'] + f"_bs{batch_size}_lr{lr}"
                        current_config['train_p'] = train_index
                        current_config['valid_p'] = valid_index
                        current_config['batch_size'] = batch_size
                        current_config['lr'] = lr
                        model = self.model(config=current_config)
                        print("====" * 25)
                        # Initialize new early stopper object for each model
                        early_stopper = EarlyStopping(mode='max', patience=self.config['patience'])
                        metrics = model.fit(dataset=dataset, epoch_end_hook=epoch_end_hook, early_stopper=early_stopper)
                        print(json.dumps(metrics['test'], indent=2, cls=CustomJsonEncoder))
                        if 'best_metrics' in self.best_params:
                            if self.best_params['best_metrics']['test']['accuracy'] < metrics['test']['accuracy']:
                                self.best_params['best_metrics'] = metrics
                        else:
                            self.best_params['best_metrics'] = metrics
                        print(f'Percentage completed: {self.global_counter / total_runs * 100}')
                        print("====" * 25, "\n\n")
            print("Best Metrics: \n")
            print("Results: ", self.best_params['best_metrics']['test'])
            del self.best_params['best_metrics']['config']['train_p'], self.best_params['best_metrics']['config'][
                'test_p']
            print("Config: ", json.dumps(self.best_params['best_metrics']['config'], indent=2, cls=CustomJsonEncoder))
            json.dump(self.best_params, open(DIFFLR_EXPERIMENTS_RUNS_PATH + f'/tuner_{self.timestamp}.json', 'w'),
                      cls=CustomJsonEncoder, indent=2)
