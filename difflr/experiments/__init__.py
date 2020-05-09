from copy import deepcopy
from sklearn.model_selection import KFold
import json
from difflr.utils import CustomJsonEncoder

class Tuner():
    def __init__(self, config, model):
        self.config= config
        self.model = model
        self.best_params = {}

    def tune(self, dataset, cv_split=3, epoch_end_hook=lambda x: x):
        kf = KFold(n_splits=cv_split)
        for train_index, test_index in kf.split(list(range(60000))):
            for epoch in self.config["epochs"]:
                for batch_size in self.config['batch_size']:
                    for lr in self.config['lr']:
                        current_config = deepcopy(self.config)
                        current_config['model_name'] = current_config['model_name']+f"_bs{batch_size}_lr{lr}"
                        current_config['train_p'] = train_index
                        current_config['test_p'] = test_index
                        current_config['batch_size'] = batch_size
                        current_config['lr'] = lr
                        current_config['epochs'] = epoch
                        model = self.model(config=current_config)
                        print("===="*25)
                        print(current_config)
                        print(model)
                        metrics = model.fit(dataset=dataset, epoch_end_hook=epoch_end_hook)
                        print(json.dumps(metrics['test_metrics'], indent=2, cls=CustomJsonEncoder))
                        print("====" * 25, "\n\n")
                        exit()

