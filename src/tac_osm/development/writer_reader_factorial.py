"""Separate MTSK transition parameters from internal read query projections."""
import hashlib
import torch

ARMS = ("writer_train_reader_train","writer_frozen_reader_train",
        "writer_train_reader_frozen","writer_frozen_reader_frozen")


def is_reader(name):
    return name.startswith(("query.","query_prior."))


def configure(model,arm):
    if arm not in ARMS:
        raise ValueError("unregistered writer/reader intervention")
    writer_train = arm.startswith("writer_train_")
    reader_train = arm.endswith("_reader_train")
    for name,p in model.memory.named_parameters():
        p.requires_grad_(reader_train if is_reader(name) else writer_train)
    return model


def group_hash(model,reader):
    digest = hashlib.sha256()
    for name,value in model.memory.state_dict().items():
        if is_reader(name)==reader:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def gradient_present(model,reader):
    return any(p.grad is not None and torch.isfinite(p.grad).all() and bool(p.grad.abs().sum()>0)
               for name,p in model.memory.named_parameters() if is_reader(name)==reader)
