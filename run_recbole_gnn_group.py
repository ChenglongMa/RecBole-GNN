import argparse
import os

from recbole_gnn.quick_start import run
from recbole.utils import list_to_latex

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_list", "-m", type=str, default="BPR", help="name of models"
    )
    parser.add_argument(
        "--dataset", "-d", type=str, default="ml-100k", help="name of datasets"
    )
    parser.add_argument("--config_files", type=str, default=None, help="config files")
    parser.add_argument(
        "--valid_latex", type=str, default="./latex/valid.tex", help="config files"
    )
    parser.add_argument(
        "--test_latex", type=str, default="./latex/test.tex", help="config files"
    )
    parser.add_argument(
        "--nproc", type=int, default=1, help="the number of process in this group"
    )
    parser.add_argument(
        "--ip", type=str, default="localhost", help="the ip of master node"
    )
    parser.add_argument(
        "--port", type=str, default="5678", help="the port of master node"
    )
    parser.add_argument(
        "--world_size", type=int, default=-1, help="total number of jobs"
    )
    parser.add_argument(
        "--group_offset",
        type=int,
        default=0,
        help="the global rank offset of this group",
    )

    args, _ = parser.parse_known_args()

    model_list = args.model_list.strip().split(",")
    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )
    dataset = args.dataset

    valid_file = f"./latex/{dataset}-valid.tex"
    test_file = f"./latex/{dataset}-test.tex"

    valid_result_list = []
    test_result_list = []

    run_times = len(model_list)

    for idx in range(run_times):
        model = model_list[idx]

        valid_res_dict = {"Model": model}
        test_res_dict = {"Model": model}
        result = run(
            model,
            args.dataset,
            config_file_list=config_file_list,
            nproc=args.nproc,
            world_size=args.world_size,
            ip=args.ip,
            port=args.port,
            group_offset=args.group_offset,
        )
        print(f'{model} trained on {args.dataset} Done!')
        valid_res_dict.update(result["best_valid_result"])
        test_res_dict.update(result["test_result"])
        bigger_flag = result["valid_score_bigger"]
        subset_columns = list(result["best_valid_result"].keys())

        valid_result_list.append(valid_res_dict)
        test_result_list.append(test_res_dict)

    df_valid, tex_valid = list_to_latex(
        convert_list=valid_result_list,
        bigger_flag=bigger_flag,
        subset_columns=subset_columns,
    )
    df_test, tex_test = list_to_latex(
        convert_list=test_result_list,
        bigger_flag=bigger_flag,
        subset_columns=subset_columns,
    )
    os.makedirs(os.path.dirname(valid_file), exist_ok=True)

    with open(valid_file, "w") as f:
        f.write(tex_valid)
    with open(test_file, "w") as f:
        f.write(tex_test)
