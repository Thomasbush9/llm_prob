"""Training step + loop with masked cross-entropy over the sample region."""

import torch
import torch.nn.functional as F


def train_step(model, opt, toks, mask, clip=1.0):
    logits = model(toks[:, :-1])
    targets = toks[:, 1:]
    shifted_mask = mask[:, 1:].float()

    loss_per_tok = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction='none',
    )
    loss = (loss_per_tok * shifted_mask.reshape(-1)).sum() / shifted_mask.sum()

    opt.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step()
    return float(loss.item()), float(grad_norm)


def train(model, loader, *, n_epochs, lr=3e-4, weight_decay=0.0,
          log_every=100, eval_every=500, eval_fn=None,
          clip=1.0, device='cpu'):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_losses, eval_history = [], []
    step = 0

    for epoch in range(n_epochs):
        for toks, mask in loader:
            toks, mask = toks.to(device), mask.to(device)
            loss, gn = train_step(model, opt, toks, mask, clip=clip)

            if step % log_every == 0:
                print(f"step {step:5d}  loss {loss:.4f}  grad_norm {gn:.3f}")
                train_losses.append({'step': step, 'loss': loss, 'grad_norm': gn})

            if eval_fn is not None and step % eval_every == 0 and step > 0:
                rec = eval_fn(model, step)
                if rec is not None:
                    eval_history.append(rec)

            step += 1

    return {'train_losses': train_losses, 'eval_history': eval_history}
