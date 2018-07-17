import argparse
import tensorflow as tf

from PspNetRunner import PspNetRunner
from PascalVocData import PascalVocData

"""
This script defines hyperparameters.
"""
def get_arguments():
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")

    # 数据
    parser.add_argument("--batch_size", type=int, default=2, help="Number of images sent to the network in one step.")
    parser.add_argument("--test_num_steps", type=int, default=1449, help="Number of training steps.")
    parser.add_argument("--data_path", type=str, default="data/pascal_train.tfrecords")
    parser.add_argument("--ignore_label", type=int, default=255, help="The index of the label to ignore during the training.")
    parser.add_argument("--input_size", type=int, default=480, help="Comma-separated string with height and width of images.")
    parser.add_argument("--num_classes", type=int, default=21, help="Number of classes to predict (including background).")

    # 训练
    parser.add_argument("--is_training", default=True, help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--decay_steps", type=int, default=15000, help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--decay_rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum component of the optimiser.")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Regularisation parameter for L2-loss.")
    parser.add_argument("--update_mean_var", default=True, help="whether to get update_op from tf.Graphic_Keys")
    parser.add_argument("--train_beta_gamma", default=True, help="whether to train beta & gamma in bn layer")
    parser.add_argument("--num_steps", type=int, default=100001, help="Number of training steps.")

    # 预测
    parser.add_argument("--output_dir", type=str, default="outputs", help="Where to save snapshots of the model.")
    parser.add_argument("--predict_data_path", type=str, default="input")

    # 模型
    parser.add_argument("--not_restore_last", default=True, help="Whether to not restore last (FC) layers.")
    parser.add_argument("--save_pred_every", type=int, default=1000, help="Save summaries and checkpoint every often.")
    parser.add_argument("--checkpoints_path", type=str, default='checkpoints/', help="Where to save snapshots of the model.")
    parser.add_argument("--logdir", type=str, default="logs", help="Where to summary of the model.")

    # 数据增强
    parser.add_argument("--random_mirror", default=True, help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random_scale", default=True, help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random_seed", type=int, default=1234, help="Random seed to have reproducible results.")

    return parser.parse_args()


def main(_):

    parser = argparse.ArgumentParser()
    parser.add_argument('--option', dest='option', type=str, default='train', help='actions: train, test, or predict')
    args = parser.parse_args()

    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    if args.option not in ['train', 'test', 'predict']:
        print('invalid option: ', args.option)
        print("Please input a option: train, test, or predict")
    else:
        data = PascalVocData(conf=get_arguments())
        # Run
        model = PspNetRunner(sess, data, config=data.conf)
        getattr(model, args.option)()


if __name__ == '__main__':
    # Choose which gpu or cpu to use
    tf.app.run()
