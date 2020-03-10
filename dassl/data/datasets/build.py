from dassl.utils import Registry, check_availability

DATASET_REGISTRY = Registry('DATASET')


def build_dataset(cfg):
    avai_datasets = DATASET_REGISTRY.registered_names()
    check_availability(cfg.DATASET.NAME, avai_datasets)
    if cfg.VERBOSE:
        print('Loading dataset: {}'.format(cfg.DATASET.NAME))
    return DATASET_REGISTRY.get(cfg.DATASET.NAME)(cfg)
