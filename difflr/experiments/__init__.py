from copy import deepcopy

class Tuner():
    def __init__(self, config, model):
        self.config= config
        self.model = model
        self.best_params = {}

    def tune(self, dataset, epoch_end_hook=None):
        # Write CV Code Here Sklearn KFold and torch Data.Subset
        for epoch in self.config["epoch"]:
            for batch_size in self.config['batch_size']:
                for lr in self.config['lr']:
                    current_config = deepcopy(self.config)
                    current_config['model_name'] = current_config['model_name']+f"_bs{batch_size}_lr{lr}"
                    model = self.model()
                    metrics = model.fit(dataset=dataset, epoch_end_hook=epoch_end_hook)
                # if metrics['']




import numpy as np
from sklearn.model_selection import KFold
X = np.zeros(100)
y = np.zeros(100)
kf = KFold(n_splits=10)
kf.get_n_splits(X)

print(kf)

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]