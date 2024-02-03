import argparse
import logging
from anyasand import Trainer

formatter = '%(asctime)s [%(name)s] %(levelname)s :  %(message)s'
logging.basicConfig(level=logging.INFO, format=formatter)
logger = logging.getLogger("anyasand-trainer")


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--in_db_path', help='input training corpus path', default="./anya-corpus.db")
    arg_parser.add_argument('-d', '--db_path', help='dictionary database path', default="./anya-dic.db")
    arg_parser.add_argument('-o', '--out_path', help='output model file path', default="anya.mdl")
    arg_parser.add_argument('-e', '--epoch_num', help='epoch num', default=100)
    args = arg_parser.parse_args()

    trainer = Trainer(args.in_db_path, args.db_path, args.epoch_num)
    trainer(args.out_path)


if __name__ == "__main__":
    main()
