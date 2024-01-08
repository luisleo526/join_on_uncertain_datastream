import logging
from argparse import ArgumentParser
from datetime import datetime

from utils import generate_time_streams

time_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh = logging.FileHandler(f'static_log_{time_string}.txt', encoding='utf-8')
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.setLevel(logging.INFO)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--num_objects', type=int, default=100)
    parser.add_argument('--num_streams', type=int, default=2)
    parser.add_argument('--dims', type=int, default=[2, 3, 4, 5, 6, 7, 8, 9, 10], nargs='+')
    parser.add_argument('--hidden_size', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--ckpt_type', choices=['best', 'last'], default='last')
    parser.add_argument('--cmt', type=str, default='precision')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    for dim in args.dims:
        streams = generate_time_streams(args.num_objects, dim, args.num_streams)
