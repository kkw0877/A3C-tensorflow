import sys
import argparse

import tensorflow as tf

import train

FLAGS = None


def add_arguments(parser):
    """Build ArgumentParser."""
    # TODO. add argument explanation
    
    # Environment
    parser.add_argument("--env", default="BreakoutDeterministic-v4", type=str)

    # Initializer
    parser.add_argument("--w_init_op", default="glorot_uniform", type=str,
                        help="truncated_normal | glorot_uniform")
    parser.add_argument("--b_init_op", default="constant", type=str)
    
    parser.add_argument("--mean", default=0.0, type=float)
    parser.add_argument("--stddev", default=0.1, type=float)
    parser.add_argument("--bias_start", default=0.0, type=float)
    
    # Learning
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--discount_factor", default=0.99, type=float)
    
    parser.add_argument("--optimizer", default="rmsprop", type=str)
    parser.add_argument("--max_gradient_norm", default=40, type=float)
    
    # Network
    parser.add_argument("--cv_num_outputs", default=16, type=int)
    parser.add_argument("--f_height", default=8, type=int)
    parser.add_argument("--f_width", default=8, type=int)
    parser.add_argument("--stride", default=4, type=int)
    parser.add_argument("--padding", default="SAME", type=str)
    
    parser.add_argument("--rnn_num_layers", default=1, type=int)
    parser.add_argument("--cell_type", default="lstm", type=str)
    parser.add_argument("--num_units", default=256, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--residual_connect", default=False, type=bool)
    
    # Model
    parser.add_argument("--model_name", default="policy_gradient", type=str,
                        help="policy_gradient | a2c")
    
    # Misc
    parser.add_argument("--out_dir", default="rmsprop", type=str)
    parser.add_argument("--num_train_steps", default=10000000, type=int)
    parser.add_argument("--steps_per_infer", default=500, type=int)
    
    parser.add_argument("--num_keep_ckpts", default=5, type=int)
    parser.add_argument("--num_gpus", default=0, type=int)
    parser.add_argument("--random_seed", default=None, type=int)

    parser.add_argument("--img_height", default=84, type=int)
    parser.add_argument("--img_width", default=84, type=int)


def main(unused_argv):
    train_fn = train.train
    train_fn(FLAGS)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
