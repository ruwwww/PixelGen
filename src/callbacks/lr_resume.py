from lightning.pytorch.callbacks import Callback

class LRResyncCallback(Callback):
    """Adjust LR schedulers on resume/start.

    - Sets scheduler.last_epoch to trainer.global_step so learning rate reflects progressed steps.
    - Optionally updates scheduler.T_max to trainer.max_steps (useful for cosine schedulers that baked in T_max).

    Add to config under trainer.callbacks:
      - class_path: src.callbacks.lr_resume.LRResyncCallback
        init_args:
          adjust_t_max: True

    """
    def __init__(self, adjust_t_max: bool = False):
        self.adjust_t_max = adjust_t_max

    def on_train_start(self, trainer, pl_module) -> None:
        gs = getattr(trainer, 'global_step', 0)
        # Lightning keeps scheduler objects in trainer.lr_scheduler_configs (list of dicts)
        scheds = getattr(trainer, 'lr_scheduler_configs', None)
        if not scheds:
            return

        for cfg in scheds:
            sched = cfg.get('scheduler', None)
            if sched is None:
                continue
            try:
                # align scheduler epoch/step with resumed global_step
                sched.last_epoch = gs
                print(f"[LRResyncCallback] Set last_epoch={gs} for scheduler: {sched.__class__.__name__}")
            except Exception as e:
                print(f"[LRResyncCallback] Failed to set last_epoch for {sched}: {e}")

            if self.adjust_t_max:
                if hasattr(sched, 'T_max'):
                    try:
                        sched.T_max = getattr(trainer, 'max_steps', sched.T_max)
                        print(f"[LRResyncCallback] Set T_max={sched.T_max} for scheduler: {sched.__class__.__name__}")
                    except Exception as e:
                        print(f"[LRResyncCallback] Failed to set T_max for {sched}: {e}")
