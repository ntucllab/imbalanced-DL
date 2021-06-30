import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import imbalanceddl.net as backbone

model_names = sorted(name for name in backbone.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(backbone.__dict__[name]))


class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


class Network(nn.Module):
    def __init__(self, cfg):
        super(Network, self).__init__()
        # config
        self.cfg = cfg
        self.num_classes = self._get_num_classes()
        self.feature_len = self._get_feature_len()
        self.backbone = self._get_backbone()
        self.classifier = self._get_classifier()

    def forward(self, x, **kwargs):
        hidden = self.backbone(x)
        out = self.classifier(hidden)
        return out, hidden

    def _get_feature_len(self):
        if self.cfg.backbone == 'resnet32':
            return 64
        elif self.cfg.backbone == 'resnet18':
            return 512
        else:
            raise ValueError("[Warning] Backbone not supported !")

    def _get_num_classes(self):
        if self.cfg.dataset == 'cifar10' or self.cfg.dataset == 'cinic10' \
                or self.cfg.dataset == 'svhn10':
            return 10
        elif self.cfg.dataset == 'cifar100':
            return 100
        elif self.cfg.dataset == 'tiny200':
            return 200
        else:
            raise NotImplementedError

    def _get_backbone(self):
        if self.cfg.backbone is not None:
            print("=> Initializing backbone : {}".format(self.cfg.backbone))
            my_backbone = backbone.__dict__[self.cfg.backbone]()
            return my_backbone
        else:
            raise ValueError("=> No backbone is specified !")

    def _get_classifier(self):
        if self.cfg.classifier is not None:
            if self.cfg.strategy == 'LDAM_DRW':
                print("=> Due to LDAM, change classifier to \
                    cosine similarity classifier !")
                self.cfg.classifier = 'cosine_similarity_classifier'
            print("=> Initializing classifier: {}".format(self.cfg.classifier))
            if self.cfg.classifier == 'dot_product_classifier':
                return nn.Linear(self.feature_len,
                                 self.num_classes,
                                 bias=False)
            elif self.cfg.classifier == 'cosine_similarity_classifier':
                return NormedLinear(self.feature_len, self.num_classes)
            else:
                raise NotImplementedError
        else:
            raise ValueError("=> No classifier is specified !")


def build_model(cfg):
    model = Network(cfg)

    if cfg.gpu is not None:
        print("=> Use GPU {} for training".format(cfg.gpu))
        torch.cuda.set_device(cfg.gpu)
        model = model.cuda(cfg.gpu)
    else:
        print("=> Use DataParallel for training")
        model = torch.nn.DataParallel(model).cuda()
    return model
