import argparse
from os import path as osp
from omegaconf import OmegaConf


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument(
        '--force_yml', nargs='+', default=None, help='Force to update yml files. Examples: train.ema_decay=0.999')
    parser.add_argument('--output_path', type=str, required=True, help='Path to option YAML file.')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_options()

    opt = OmegaConf.load(args.base_opt)
    # force to update yml options
    if args.force_yml is not None:
        for entry in args.force_yml:
            # now do not support creating new keys
            keys, value = entry.split('=')
            keys, value = keys.strip(), value.strip()
  
            if "<TOK>" in keys:
                keys = keys.split('.<TOK>')[0]
                cmd = f"opt.{keys}['<TOK>'] = '{value}'"
                exec(cmd)
            else:
                exec(f"print(opt.{keys})")
                try:
                    cmd = f"opt.{keys} = {value}"
                    exec(cmd)
                except:
                    cmd = f"opt.{keys} = '{value}'"
                    exec(cmd)

                print(cmd)
                exec(f"print(opt.{keys})")
            print('\n')

    # save updated yml
    OmegaConf.save(opt, args.output_path)
