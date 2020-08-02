import argparse
import logging
from train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script.')
    parser.add_argument('--mode', dest='mode', type=str, default='train')
    parser.add_argument('--logging_level', dest='logging_level', type=str, default='info')
    parser.add_argument('--render', dest='render', action='store_true', default=False)
    parser.add_argument('--max_eps', dest='max_eps', type=int, default=1000)
    parser.add_argument('--max_steps', dest='max_steps', type=int, default=2000)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=256)
    args = parser.parse_args()
    if args.logging_level == 'critical':
        logging.basicConfig(level=logging.CRITICAL)
    if args.logging_level == 'info':
        logging.basicConfig(level=logging.INFO)
    if args.mode == 'train':
        train(render=args.render, max_eps=args.max_eps, max_steps=args.max_steps, batch_size=args.batch_size)
    else:
        raise RuntimeError('unknown_mode')