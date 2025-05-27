def Trainer(trainer_cls):
    def decorator(cls):
        cls.trainer_cls = trainer_cls
        return cls
    return decorator