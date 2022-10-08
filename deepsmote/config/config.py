import argparse

def get_args():
    parser = argparse.ArgumentParser(
        add_help=False, description='PyTorch Deep Imbalanced Training')

    parser.add_argument('--dim_h', default=64, type=int, help='factor controlling size of hidden layers')
    parser.add_argument('--n_channel', default=3, type=int, help='number of channels in the input data')
    parser.add_argument('--n_z', default=600, type=int, help='number of dimensions in latent space')
    parser.add_argument('--sigma', default=1.0, type=float, help='variance in n_z')
    parser.add_argument('--lambda', default=0.01, type=float, help='hyper param for weight of discriminator loss')
    parser.add_argument('--lr', default=0.0002, type=float, metavar='LR', help='learning rate for Adam optimizer .000')
    parser.add_argument('--epochs', default=200, type=int, help='how many epochs to run for')

    parser.add_argument('--batch_size', default=128, type=int, help='batch size for SGD')
    parser.add_argument('--save', action='store_false', help='save weights at each epoch of training if True')
    parser.add_argument('--train', action='store_false', help='save weights at each epoch of training if True')
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset to use')

    parser.add_argument('--imb_type', default='exp', help='type of imbalance')
    parser.add_argument('--imb_factor', default=0.01, type=float, help='ratio of imbalance')

    args = parser.parse_args()

    return args