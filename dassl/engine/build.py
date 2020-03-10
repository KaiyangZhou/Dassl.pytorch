from dassl.utils import Registry, check_availability

TRAINER_REGISTRY = Registry('TRAINER')


def build_trainer(cfg):
    avai_trainers = TRAINER_REGISTRY.registered_names()
    check_availability(cfg.TRAINER.NAME, avai_trainers)
    if cfg.VERBOSE:
        print('Loading trainer: {}'.format(cfg.TRAINER.NAME))
    return TRAINER_REGISTRY.get(cfg.TRAINER.NAME)(cfg)
