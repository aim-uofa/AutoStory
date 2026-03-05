import argparse
from os import path as osp
import glob
from omegaconf import OmegaConf


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--potential_dir', type=str, required=True)
    parser.add_argument('--depth', type=int, required=True)
    parser.add_argument('--file_key', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None, required=True)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_options()

    potential_dir = args.potential_dir
    for _ in range(args.depth):
        potential_dir = osp.join(potential_dir, '*')
    
    if args.file_key is not None:
        potential_dir = osp.join(potential_dir, f"*{args.file_key}*")

    path_list = []
    for potential_path in glob.glob(potential_dir):
        path_list.append(potential_path)
    
    # save to txt
    with open(args.save_path, 'w') as f:
        for path in path_list:
            f.write(f"{path}\n")
