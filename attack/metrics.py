import torch
from collections import defaultdict

def total_variation(x):
    """Anisotropic TV."""
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy

def psnr(img_batch, ref_batch, batched=False, factor=1.0):
    """Standard PSNR."""
    def get_psnr(img_in, img_ref):
        mse = ((img_in - img_ref)**2).mean()
        if mse > 0 and torch.isfinite(mse):
            return (10 * torch.log10(factor**2 / mse))
        elif not torch.isfinite(mse):
            return img_batch.new_tensor(float('nan'))
        else:
            return img_batch.new_tensor(float('inf'))

    if batched:
        psnr = get_psnr(img_batch.detach(), ref_batch)
    else:
        [B, C, m, n] = img_batch.shape
        psnrs = []
        for sample in range(B):
            psnrs.append(get_psnr(img_batch.detach()[sample, :, :, :], ref_batch[sample, :, :, :]))
        psnr = torch.stack(psnrs, dim=0).mean()

    return psnr.item()

def activation_errors(model, x1, x2):
    """Compute activation-level error metrics for every module in the network."""
    model.eval()

    device = next(model.parameters()).device

    hooks = []
    data = defaultdict(dict)
    inputs = torch.cat((x1, x2), dim=0)
    separator = x1.shape[0]

    def check_activations(self, input, output):
        module_name = str(*[name for name, mod in model.named_modules() if self is mod])
        try:
            layer_inputs = input[0].detach()
            residual = (layer_inputs[:separator] - layer_inputs[separator:]).pow(2)
            se_error = residual.sum()
            mse_error = residual.mean()
            sim = torch.nn.functional.cosine_similarity(layer_inputs[:separator].flatten(),
                                                        layer_inputs[separator:].flatten(),
                                                        dim=0, eps=1e-8).detach()
            data['se'][module_name] = se_error.item()
            data['mse'][module_name] = mse_error.item()
            data['sim'][module_name] = sim.item()
        except (KeyboardInterrupt, SystemExit):
            raise
        except AttributeError:
            pass

    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(check_activations))

    try:
        outputs = model(inputs.to(device))
        for hook in hooks:
            hook.remove()
    except Exception as e:
        for hook in hooks:
            hook.remove()
        raise

    return data