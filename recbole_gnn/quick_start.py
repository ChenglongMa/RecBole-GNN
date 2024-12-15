import os
import sys

import pandas as pd
import time
import logging
from collections.abc import MutableMapping
from logging import getLogger
from recbole.utils import init_logger, init_seed, set_color

from recbole_gnn.config import Config
from recbole_gnn.utils import create_dataset, data_preparation, get_model, get_trainer
import torch.multiprocessing as mp
import torch


def is_windows():
    return sys.platform in ["win32", "cygwin"]


def run(
    model,
    dataset,
    config_file_list=None,
    config_dict=None,
    saved=True,
    nproc=1,
    world_size=-1,
    ip="localhost",
    port="5678",
    group_offset=0,
):
    if nproc == 1 and world_size <= 0:
        mp.set_sharing_strategy("file_system")

        config_dict = config_dict or {}
        config_dict["worker"] = 0 if is_windows() else os.cpu_count()
        print(f'Using {config_dict["worker"]} processes')

        res = run_recbole_gnn(
            model=model,
            dataset=dataset,
            config_file_list=config_file_list,
            config_dict=config_dict,
            saved=saved,
        )
    else:
        nproc = nproc if nproc > 0 else torch.cuda.device_count()
        print(f"Using {nproc} processes")

        if world_size == -1:
            world_size = nproc

        # Refer to https://discuss.pytorch.org/t/problems-with-torch-multiprocess-spawn-and-simplequeue/69674/2
        # https://discuss.pytorch.org/t/return-from-mp-spawn/94302/2
        queue = mp.get_context("spawn").SimpleQueue()

        config_dict = config_dict or {}
        config_dict.update(
            {
                "world_size": world_size,
                "ip": ip,
                "port": port,
                "nproc": nproc,
                "offset": group_offset,
                "worker": 0,
                "gpu_id": ",".join([str(i) for i in range(nproc)]),
            }
        )
        kwargs = {
            "config_dict": config_dict,
            "queue": queue,
        }
        mp.spawn(
            run_recbole_gnns,
            args=(model, dataset, config_file_list, kwargs),
            nprocs=nproc,
            join=True,
        )
        print(f"Training Done")
        # Normally, there should be only one item in the queue
        res = None if queue.empty() else queue.get()
        print("Collect Result")
    print("Done")
    return res


def run_recbole_gnn(
    model=None,
    dataset=None,
    config_file_list=None,
    config_dict=None,
    saved=True,
    queue=None,
):
    r"""A fast running api, which includes the complete process of
    training and testing a model on a specified dataset
    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
        queue (torch.multiprocessing.Queue, optional): The queue used to pass the result to the main process. Defaults to ``None``.
    """
    # configurations initialization
    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    try:
        assert config["enable_sparse"] in [True, False, None]
    except AssertionError:
        raise ValueError("Your config `enable_sparse` must be `True` or `False` or `None`")
    init_seed(config["seed"], config["reproducibility"])
    # logger initialization
    if config["local_rank"] == 0:
        init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data.dataset).to(config["device"])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config["show_progress"]
    )

    # model evaluation
    test_result = trainer.evaluate(
        test_data, load_best_model=saved, show_progress=config["show_progress"]
    )
    if "topk_results" in test_result:
        topk_results = test_result.pop("topk_results")
        save_results(config, test_result, topk_results)

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    result = {
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
        # "topk_results": topk_results,
    }

    if config["local_rank"] == 0 and queue is not None:
        print(f"Return result to mp.spawn")
        queue.put(result)
        print(f"Return result to mp.spawn Done")

    if not config["single_spec"]:
        import torch.distributed as dist

        dist.destroy_process_group()
    return result


def run_recbole_gnns(rank, *args):
    kwargs = args[-1]
    if not isinstance(kwargs, MutableMapping):
        raise ValueError(
            f"The last argument of run_recboles should be a dict, but got {type(kwargs)}"
        )
    kwargs["config_dict"] = kwargs.get("config_dict", {})
    kwargs["config_dict"]["local_rank"] = rank
    run_recbole_gnn(
        *args[:3],
        **kwargs,
    )


def save_results(config, test_result, topk_results):
    if config["local_rank"] != 0:
        return
    print(f"Saving Result...")
    now = time.strftime("%y%m%d%H%M%S")
    eval_results = []
    model_name = config["model"]
    dataset_name = config["dataset"]

    for metric, value in test_result.items():
        eval_results.append([model_name, metric, value])
    eval_results = pd.DataFrame(eval_results, columns=["model", "metric", "value"])
    result_dir = config["result_dir"]
    os.makedirs(result_dir, exist_ok=True)

    nproc = config["nproc"]
    filename = os.path.join(
        result_dir, f"result_{model_name}_{dataset_name}_{now}_{nproc}.csv"
    )
    if os.path.exists(filename):
        print(f"{filename} already exists!")
    else:
        eval_results.to_csv(filename, index=False)

    filename = os.path.join(
        result_dir, f"topk_{model_name}_{dataset_name}_{now}_{nproc}.csv"
    )
    if os.path.exists(filename):
        print(f"{filename} already exists!")
    else:
        topk_results.to_csv(filename, index=False)
    print(f"Saving Result Done!")


def objective_function(config_dict=None, config_file_list=None, saved=True):
    r"""The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    try:
        assert config["enable_sparse"] in [True, False, None]
    except AssertionError:
        raise ValueError("Your config `enable_sparse` must be `True` or `False` or `None`")
    init_seed(config["seed"], config["reproducibility"])
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data.dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, verbose=False, saved=saved
    )
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    return {
        'model': config['model'],
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }
