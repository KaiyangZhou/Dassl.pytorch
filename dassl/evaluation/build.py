from dassl.utils import Registry, check_availability

EVALUATOR_REGISTRY = Registry('EVALUATOR')


def build_evaluator(cfg, **kwargs):
    avai_evaluators = EVALUATOR_REGISTRY.registered_names()
    check_availability(cfg.TEST.EVALUATOR, avai_evaluators)
    if cfg.VERBOSE:
        print('Loading evaluator: {}'.format(cfg.TEST.EVALUATOR))
    return EVALUATOR_REGISTRY.get(cfg.TEST.EVALUATOR)(cfg, **kwargs)
