"""GPU-resident parameters with FP32 Adam moments staged one tensor at a time.

Uses PyTorch AdamW unchanged for each independent parameter update. Only moment
placement changes; FSDP parameters and accumulated gradients stay on the GPU.
The generic engine optimizer_offload flag must remain false: it stages all
moments together and defeats this optimizer's bounded staging.
"""
import torch


class StreamingAdamW(torch.optim.AdamW):
    def __init__(self, params, **kwargs):
        for key in ('foreach', 'fused', 'capturable', 'differentiable'):
            if kwargs.get(key, False):
                raise ValueError(f'{key}=true is unsupported by StreamingAdamW')
        kwargs.update(foreach=False, fused=False, capturable=False, differentiable=False)
        super().__init__(params, **kwargs)

    def load_state_dict(self, state_dict):
        # AdamW restores moments onto each parameter's device. Move them back
        # before the resumed forward pass, preserving bounded GPU staging.
        result = super().load_state_dict(state_dict)
        for state in self.state.values():
            for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                if key in state:
                    state[key] = state[key].to('cpu')
        return result

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        groups = self.param_groups
        try:
            for group in groups:
                for param in group['params']:
                    if param.grad is None:
                        continue
                    if param.dtype != torch.float32:
                        raise ValueError('StreamingAdamW requires FP32 parameter shards')
                    state = self.state[param]
                    for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                        if key in state:
                            state[key] = state[key].to(param.device)
                    self.param_groups = [dict(group, params=[param])]
                    try:
                        super().step()
                    finally:
                        # Blocking copies guarantee the CPU moments are ready on
                        # return; do not leave a second GPU tensor's states live.
                        for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                            if key in state:
                                state[key] = state[key].to('cpu')
        finally:
            self.param_groups = groups
        return loss
