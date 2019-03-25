# Experiment Runner (exp-runner)

**exp-runner** is a simple and extensible framework for data analysis and machine learning experiments in Python.

#### Structure
The framework includes following step:
1. _Data loading_
2. _Data transformation_
3. _Model training and testing_
4. _Performance evaluation_
5. _Results saving_

#### Main features
* _Generability_: Variaty of models and methods are supported and it can be used in a number of tasks 
(such as preprocessing, dimensionality reduction, classification, 
regression, clustering, statistical tests, etc.)
* _Flexability_: Steps can be easily skipped and/or included
* _Dynamic loading_: Automatically imports modules during runtime - no additional lines are needed

### Installation
```bash
pip install exp-runner
```

### Usage 

Let's say, your project has the following structure:
    
    MyAwesomeProject/
            main.py
            my_custom_module.py
            
            data/
                    data_00.npy
                    data_01.npy
                    ...
                    data_NN.npy
            
            protocols/
                    experiment_config.json
            
            results/
            

#### Just give me a code!
You just need to describe your **framework** in the [JSON](https://json.org) configuration file:

##### experiment_config.json
```JSON
{
  "Setup": {
    "description": "You can add detailed description of the experiment",
    "random_seed": 42
  },
  "Dataset": {
    "class": "my_custom_module.MyAwesomeDataLoader",
    "args": {"path_to_data": "data/*.npy"}
  },
  "Transforms": [
    {
      "class": "sklearn.decomposition.PCA",
      "args": {"n_components": 3, "whiten": true}
    }
  ],
  "Model": {
    "class": "sklearn.cluster.KMeans",
    "args": {"n_clusters": 3, "n_jobs": -1, "verbose": 0}
  },
  "Metric": {
    "class": "my_custom_module.SklearnMetricWrapper",
    "args": {"metric": "normalized_mutual_info_score"}
  },
  "Saver": {
    "class": "my_custom_module.CSVReport",
    "args": {"path_to_output": "results/evaluation_results.csv", "sep": ";"}
  }
}
```
<details><summary>Here are aforementioned classes (click):</summary>
<p>

##### my_custom_module.py

```python
import os
import glob
import numpy as np
import sklearn.metrics

from exp_runner import Dataset, Metric, Saver

from collections import defaultdict
from typing import Any, Dict, List, Union, NoReturn, Iterable, Callable

from sklearn.model_selection import StratifiedShuffleSplit


class MyAwesomeDataLoader(Dataset):

    def __init__(self, path_to_data: str, test_size: float = 0.1, training: bool = True):

        super(MyAwesomeDataLoader, self).__init__()

        self._samples = dict()
        self._labels = dict()
        self._splits = defaultdict(dict)

        paths_to_data = glob.glob(path_to_data)

        for path in paths_to_data:
            fname = os.path.basename(path)

            data = np.load(path)
            X = data[:, :-1]   
            y = data[:, -1]

            indices_train, indices_test = next(StratifiedShuffleSplit(
                test_size=test_size
            ).split(X, y))

            self._samples[fname] = X
            self._labels[fname] = y
            self._splits[fname]['train'] = indices_train
            self._splits[fname]['test'] = indices_test

        self._indices = list(self._samples.keys())

        self._training = training

    def __getitem__(self, index: int) -> Dict[str, Dict[str, Union[str, np.ndarray]]]:
        if not (0 <= index < len(self._indices)):
            raise IndexError

        fname = self._indices[index]

        item = {
        'X': self._samples[fname][self._splits[fname]['train'] if self.training else self._splits[fname]['test']],
        'y': self._labels[fname][self._splits[fname]['train'] if self.training else self._splits[fname]['test']]
        }

        item['desc'] = 'it is possible to add description for each data sample'

        return {'filename': fname, 'item': item}

    def __len__(self) -> int:
        return len(self._indices)

    @property
    def training(self):
        return self._training


class SklearnMetricWrapper(Metric):

    def __init__(self, metric: str):
        super(SklearnMetricWrapper, self).__init__()

        metric = getattr(sklearn.metrics, metric)
        self._metric: Callable[[Iterable[Union[float, int]], Iterable[Union[float, int]]], float] = metric

    def __call__(self, y_true: Iterable[Union[float, int]], y_pred: Iterable[Union[float, int]]) -> float:
        return self._metric(y_true, y_pred)


class CSVReport(Saver):

    def __init__(self, path_to_output: str, sep: str = ';', append: bool = True):
        super(CSVReport, self).__init__()

        self.path_to_output = path_to_output
        self.sep = sep
        self.mode = 'a+' if append else 'w+'

    def save(self, report: List[Dict[str, Any]]) -> NoReturn:
        with open(self.path_to_output, self.mode) as csv:
            for entry in report:
                line = self.sep.join([
                    entry['filename'],
                    entry['desc'],
                    entry['perf']
                ]) + '\n'
                csv.write(line)
```

</p>
</details>

Finally, to run your experiment type in your terminal:
```bash
cd /path/to/MyAwesomeProject
python main.py --config protocols/experiment_config.json
```
