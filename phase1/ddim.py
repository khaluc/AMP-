"""DDIM eta=0 inference adapter; preserves the checkpoint's design-length conditioning.

Based on the DDIM update in the official starter kit, with explicit keyword arguments.
This is an inference variant, not the published 1,000-step ancestral baseline.
"""
import torch


@torch.no_grad()
def ddim_sample(model, batch_size, design_len, steps=100):
    if not 1 <= steps <= model.num_timesteps:
        raise ValueError("DDIM steps must be between 1 and the training timestep count")
    device = model.betas.device
    x = torch.randn(batch_size, model.seq_length, model.embed_dim, device=device)
    times = torch.linspace(-1, model.num_timesteps - 1, steps + 1).long().tolist()[::-1]
    x_start = None
    for current, following in zip(times[:-1], times[1:]):
        t = torch.full((batch_size,), current, device=device, dtype=torch.long)
        predictions = model.model_predictions(
            x, t, design_len=design_len,
            x_self_cond=x_start if model.self_condition else None,
            clip_x_start=False,
        )
        x_start = predictions.pred_x_start
        if following < 0:
            x = x_start
        else:
            alpha_next = model.alphas_cumprod[following]
            x = alpha_next.sqrt() * x_start + (1 - alpha_next).sqrt() * predictions.pred_noise
    if not torch.isfinite(x).all():
        raise RuntimeError("Non-finite DDIM output")
    return model.unnormalize(x)
