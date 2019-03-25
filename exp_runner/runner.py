import os
import json
import tqdm
import argparse
import importlib
import numpy as np

from sklearn.base import ClusterMixin
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin

from typing import Any
from typing import Dict
from typing import List
from typing import Type
from typing import Union

RANDOM_SEED = 42


def load_class(package_name: str, class_name: str) -> Type:

    importlib.invalidate_caches()

    package = importlib.import_module(package_name)
    clazz = getattr(package, class_name)

    return clazz


def run(experiment_name: str,
        experiment_desc: str,
        model_class: str,
        model_args: Dict[str, Any],
        dataset_class: str,
        dataset_args: Dict[str, Any],
        transforms: List[Dict[str, Union[str, Dict[str, Any]]]],
        metric_class: str,
        metric_args: Dict[str, Any],
        saver_class: str,
        saver_args: Dict[str, Any]):

    print(f'Starting "{experiment_name}":\n-> {experiment_desc}')

    Dataset: Type = load_class(*dataset_class.rsplit('.', 1))
    dataset_train = Dataset(**{**dataset_args, **{'training': True}})
    dataset_test = Dataset(**{**dataset_args, **{'training': False}})

    data_transforms: List[Any, TransformerMixin] = list()
    for idx, transform in enumerate(transforms):
        data_transforms.append(load_class(*transform['class'].rsplit('.', 1))(**transform['args']))

    Model: Type[Any, Union[ClusterMixin, ClassifierMixin]] = load_class(*model_class.rsplit('.', 1))
    model = Model(**model_args)

    Metric: Type = load_class(*metric_class.rsplit('.', 1))
    metric = Metric(**metric_args)

    Saver: Type = load_class(*saver_class.rsplit('.', 1))
    saver = Saver(**saver_args)

    eval_results = list()
    for entry_train, entry_test in tqdm.tqdm(zip(dataset_train, dataset_test), total=len(dataset_train)):
        entry_fname: str = entry_train['filename']
        entry_desc: str = entry_train['item']['desc']

        X_train: np.ndarray = entry_train['item']['X']
        y_train: np.ndarray = entry_train['item']['y']

        X_test: np.ndarray = entry_test['item']['X']
        y_test: np.ndarray = entry_test['item']['y']

        for transform in data_transforms:
            transform = transform.fit(X_train, y_train)

            if hasattr(transform, 'transform'):
                X_train = transform.transform(X_train)
                X_test = transform.transform(X_test)
            else:
                X_train = transform.fit_transform(X_train, y_train)
                X_test = transform.fit_transform(X_test, y_test)

        model = model.fit(X_train, y_train)

        y_pred: np.ndarray = model.predict(X_test)
        performance: float = metric(y_test, y_pred)

        eval_results.append({
            'filename': entry_fname,
            'desc': entry_desc,
            'perf': str(performance),
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        })

    saver.save(eval_results)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--random-seed', type=int, required=False)

    args = parser.parse_args()

    config = json.load(open(args.config))

    experiment_name = str(os.path.splitext(os.path.basename(args.config))[0])
    experiment_desc = config['Setup']['description'] or ''

    random_seed = int(args.random_seed or config['Setup']['random_seed'] or RANDOM_SEED)
    np.random.seed(random_seed)

    model_class = config['Model']['class']
    model_args = config['Model']['args']
    dataset_class = config['Dataset']['class']
    dataset_args = config['Dataset']['args']
    transforms = config['Transforms']
    metric_class = config['Metric']['class']
    metric_args = config['Metric']['args']
    saver_class = config['Saver']['class']
    saver_args = config['Saver']['args']

    run(
        experiment_name=experiment_name,
        experiment_desc=experiment_desc,
        model_class=model_class,
        model_args=model_args,
        dataset_class=dataset_class,
        dataset_args=dataset_args,
        transforms=transforms,
        metric_class=metric_class,
        metric_args=metric_args,
        saver_class=saver_class,
        saver_args=saver_args
    )
