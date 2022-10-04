import argparse
import yaml

def get_args():
    parser = argparse.ArgumentParser(
        add_help=False, description='PyTorch Deep Imbalanced Training')

    # Load params from config file
    parser.add_argument('-c', '--config', help='Path to configuration file')
    args, _ = parser.parse_known_args()
    config = {}
    # Default settings
    if args.config:
        with open(args.config) as fp:
            config = yaml.load(fp, Loader=yaml.SafeLoader)

    # Imbalance dataset
    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset to use')
    parser.add_argument('--imb_type', default="exp", type=str, choices=['exp', 'step'], help='imbalance type')
    parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
    # Strategy
    parser.add_argument('--strategy', default="ERM", type=str,  choices=['ERM', 'DRW', 'LDAM_DRW', 'Mixup_DRW', 'Remix_DRW','Reweight_CB', 'MAMix_DRW', 'M2m', 'Deep_SMOTE'
                        ], help='select strategy for trainer')
    parser.add_argument('--learning_rate', default=0.1, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=2e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    # Seed
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')

    # M2m
    parser.add_argument('--resume', '-r', action='store_false',help='resume from checkpoint')
    parser.add_argument('--net_g', default=None, type=str, help='checkpoint path of network for generation')
    parser.add_argument('--net_g2', default=None, type=str, help='checkpoint path of network for generation')
    parser.add_argument('--net_t', default=None, type=str, help='checkpoint path of network for train')
    parser.add_argument('--net_both', default=None, type=str, help='checkpoint path of both networks')
    parser.add_argument('--backbone', default='resnet32', type=str, help='model type (default: ResNet18)')
    parser.add_argument('--effect_over', action='store_true', help='Use effective number in oversampling')
    parser.add_argument('--no_over', dest='over', action='store_false', help='Do not use over-sampling')
    parser.add_argument('--gen', '-gen', action='store_false', help='')
    parser.add_argument('--warm', default=160, type=int, help='Deferred strategy for re-balancing')
    parser.add_argument('--epochs', default=200, type=int,help='total epochs to run')
    parser.add_argument('--loss_type', default='CE', type=str, choices=['CE', 'Focal', 'LDAM'], help='Type of loss for imbalance')
    parser.add_argument('--reweight', '-reweight', action='store_true', help='oversampling')
    parser.add_argument('--no-augment', dest='augment', action='store_false', help='use standard augmentation (default: True)')
    parser.add_argument('--attack_iter', default=10, type=int, help='')
    parser.add_argument('--lam', default=0.5, type=float, help='Hyper-parameter for regularization of translation')
    parser.add_argument('--gamma', default=0.99, type=float, help='Threshold of the generation')
    parser.add_argument('--beta', default=0.999, type=float, help='Hyper-parameter for rejection/sampling')
    parser.add_argument('--step_size', default=0.1, type=float, help='')
    parser.add_argument('--smote', '-s', action='store_true', help='oversampling')

    # Log
    parser.add_argument('--root_log', type=str, default='log')
    parser.add_argument('--root_model', type=str, default='checkpoint')
    # Assign GPU
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    # Evaluation with Best Model
    parser.add_argument('--best_model', default=None, type=str, metavar='PATH', help='Path to Best Model')

    # update config from command line
    parser.set_defaults(**config)
    args = parser.parse_args()

    return args