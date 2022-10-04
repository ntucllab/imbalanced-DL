from imbalanceddl.strategy import MixupTrainer
from imbalanceddl.strategy import RemixTrainer
from imbalanceddl.strategy import MAMixTrainer
from imbalanceddl.strategy import ERMTrainer
from imbalanceddl.strategy import DRWTrainer
from imbalanceddl.strategy import LDAMDRWTrainer
from imbalanceddl.strategy import ReweightCBTrainer
from imbalanceddl.strategy import M2mTrainer
from imbalanceddl.strategy import DeepSMOTETrainer


def build_trainer(cfg, imbalance_dataset, model=None, strategy=None):
    """
    Build various strategy (trainer) specified by users
    """
    if strategy == 'Mixup_DRW':
        print("=> Mixup Trainer !")
        trainer = MixupTrainer(cfg,
                               imbalance_dataset,
                               model=model,
                               strategy=strategy)
    elif strategy == 'M2m':
        print("=> M2m Trainer !")
        trainer = M2mTrainer(cfg,
                               imbalance_dataset,
                               model=model,
                               strategy=strategy)

    elif strategy == 'Deep_SMOTE':
        print("=> Deep_SMOTE Trainer !")
        trainer = DeepSMOTETrainer(cfg,
                               imbalance_dataset,
                               model=model,
                               strategy=strategy)                                    
    elif strategy == 'Remix_DRW':
        print("=> Remix Trainer !")
        trainer = RemixTrainer(cfg,
                               imbalance_dataset,
                               model=model,
                               strategy=strategy)
    elif strategy == 'MAMix_DRW':
        print("=> MAMix Trainer !")
        trainer = MAMixTrainer(cfg,
                               imbalance_dataset,
                               model=model,
                               strategy=strategy)
    elif strategy == 'ERM':
        print("=> ERM Trainer !")
        trainer = ERMTrainer(cfg,
                             imbalance_dataset,
                             model=model,
                             strategy=strategy)
    elif strategy == 'DRW':
        print("=> DRW Trainer !")
        trainer = DRWTrainer(cfg,
                             imbalance_dataset,
                             model=model,
                             strategy=strategy)
    elif strategy == 'LDAM_DRW':
        print("=> LDAM_DRW Trainer !")
        trainer = LDAMDRWTrainer(cfg,
                                 imbalance_dataset,
                                 model=model,
                                 strategy=strategy)
    elif strategy == 'Reweight_CB':
        print("=> Reweight_CB Trainer !")
        trainer = ReweightCBTrainer(cfg,
                                    imbalance_dataset,
                                    model=model,
                                    strategy=strategy)
    else:
        raise NotImplementedError

    return trainer