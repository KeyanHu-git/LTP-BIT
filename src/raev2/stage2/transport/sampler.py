import torch as th


class Sampler:
    def __init__(self, transport, guidance_config):
        self.transport = transport
        self.drift = self.transport.get_drift()
        self.guidance_config = guidance_config
        cfg = guidance_config.get("cfg") if guidance_config is not None else None
        self.omega = cfg.get("scale") if cfg is not None else None
        self.t_start = cfg.get("t_min") if cfg is not None else None
        self.t_end = cfg.get("t_max") if cfg is not None else None

    def sample_ode(self, *, num_steps=50, capture_steps=None):
        if capture_steps is None:
            capture_set = None
        else:
            capture_steps = tuple(int(step) for step in capture_steps)
            if (
                not capture_steps
                or capture_steps != tuple(sorted(set(capture_steps)))
                or capture_steps[0] < 1
                or capture_steps[-1] > num_steps
            ):
                raise ValueError("capture_steps must be unique, increasing, and within [1, num_steps].")
            capture_set = set(capture_steps)
        t_grid = th.linspace(1.0, 0.0, num_steps + 1)
        shift = self.transport.time_dist_shift
        t_grid = shift * t_grid / (1 + (shift - 1) * t_grid)

        def sample_fn(x, model, **model_kwargs):
            device = x.device
            t_steps = t_grid.to(device)
            B = x.shape[0]
            captured = []

            model_kwargs_ = model_kwargs.copy()
            for k, v in (('omega', self.omega), ('t_start', self.t_start), ('t_end', self.t_end)):
                if v is not None:
                    model_kwargs_[k] = th.full((B,), v, device=device)

            for i in range(num_steps):
                h = t_steps[i] - t_steps[i + 1]
                t_batch = th.full((B,), t_steps[i].item(), device=device)
                d_cur = self.drift(x, t_batch, model, **model_kwargs_)
                x = x - h * d_cur
                if capture_set is not None and i + 1 in capture_set:
                    captured.append(x)

            return x.unsqueeze(0) if capture_set is None else th.stack(captured, dim=0)

        return sample_fn
