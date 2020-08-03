import argparse
import logging

from tensorflow.python.framework.dtypes import float32
from train import train, explore

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script.')
    parser.add_argument('--mode', dest='mode', type=str, default='train')
    parser.add_argument('--checkpoint_location',
                        dest='checkpoint_location', type=str, default='most_recent')
    parser.add_argument('--model_location',
                        dest='model_location', type=str, default='most_recent')
    parser.add_argument('--game_id', dest='game_id', type=str, default='Berzerk-v0',
                        help='The game we want to play. Allowed values are: Berzerk-v0 (default), CartPole-v1, Skiing-v0')
    parser.add_argument('--logging_level',
                        dest='logging_level', type=str, default='info')
    parser.add_argument('--render', dest='render',
                        action='store_true', default=False)
    parser.add_argument('--max_eps', dest='max_eps', type=int, default=1000)
    parser.add_argument('--eval_freq', dest='eval_freq', type=int, default=10)
    parser.add_argument('--train_per_episode',
                        dest='train_per_episode', type=int, default=1)
    parser.add_argument('--rounds_per_episode',
                        dest='rounds_per_episode', type=int, default=1)
    parser.add_argument('--update_freq',
                        dest='update_freq', type=int, default=3)
    parser.add_argument('--max_steps', dest='max_steps',
                        type=int, default=2000)
    parser.add_argument('--batch_size', dest='batch_size',
                        type=int, default=256)
    parser.add_argument('--epsilon', dest='epsilon', type=float, default=0.05)
    parser.add_argument('--learning_rate',
                        dest='learning_rate', type=float, default=1e-3)
    args = parser.parse_args()
    if args.logging_level == 'critical':
        logging.basicConfig(level=logging.CRITICAL)
    if args.logging_level == 'info':
        logging.basicConfig(level=logging.INFO)
    if args.mode == 'train':
        train(render=args.render, max_eps=args.max_eps,
              max_steps=args.max_steps, game_id=args.game_id,
              batch_size=args.batch_size, epsilon=args.epsilon,
              learning_rate=args.learning_rate,
              train_per_episode=args.train_per_episode,
              rounds_per_episode=args.rounds_per_episode,
              update_freq=args.update_freq,
              checkpoint_location=args.checkpoint_location,
              model_location=args.model_location,
              eval_freq=args.eval_freq)
    elif args.mode == 'explore':
        explore(game_id=args.game_id)
    else:
        raise RuntimeError('unknown_mode')
