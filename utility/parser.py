"""
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
"""

import argparse



def parse_args():
    parser = argparse.ArgumentParser(description="Run MBSSL.")

    # ******************************   optimizer paras      ***************************** #
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate."  # common parameter
    )
    parser.add_argument("--test_epoch", type=int, default=5, help="test epoch steps.")
    parser.add_argument("--data_path", nargs="?", default="../Data/", help="Input data path.")

    parser.add_argument(
        "--dataset", nargs="?", default="Taobao", help="Choose a dataset from {Taobao,Tmall}"
    )
    parser.add_argument("--verbose", type=int, default=1, help="Interval of evaluation.")
    parser.add_argument("--epoch", type=int, default=2048, help="Number of epoch.")

    parser.add_argument(
        "--embed_size", type=int, default=64, help="Embedding size."  # common parameter
    )
    parser.add_argument(
        "--depth",
        default=4,
        type=int,
        help="mlp depth for behs",
    )
    parser.add_argument(
        "--expansion_factor",
        default=2,
        type=int,
        help="expansion_factor",
    )

    parser.add_argument("--batch_size", type=int, default=1536, help="Batch size.")
    parser.add_argument("--gpu_id", type=int, default=0, help="Gpu id")

    parser.add_argument("--Ks", nargs="?", default="[10, 20, 40]", help="K for Top-K list")
    parser.add_argument("--rel_weight", default="[0.4, 0.4, 0.2]", help="behavior weight")
    parser.add_argument("--coeff", default="[1,1,0.1]", help="loss weight for different task")

    parser.add_argument(
        "--save_flag", type=int, default=0, help="0: Disable model saver, 1: Activate model saver"
    )

    parser.add_argument(
        "--test_flag",
        nargs="?",
        default="part",
        help="Specify the test type from {part, full}, indicating whether the reference is done in mini-batch",
    )

    # ******************************   model hyper paras      ***************************** #

    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()
