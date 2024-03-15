import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import Union, Any
from torch.optim import Optimizer
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from .utils import save_checkpoint

try:
    import wandb

    _has_wandb = True
except ImportError:
    _has_wandb = False


def train_step(model, data_loader, optimizer, scaler, grad_clip=None):
    model.train()
    a_loss, t_loss, a_acc = 0, 0, 0
    for batch, items in enumerate(data_loader):
        cat, num, constraints, target = (
            items["cat"],
            items["num"],
            items["constraints"],
            items["target"],
        )

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            target = target.view(-1)
            mask = target != 0

            logits, reg, _ = model(x=(cat, num), constraints=constraints, mask=mask)
            logits = logits.view(-1, logits.shape[-1])

            a_l = F.cross_entropy(logits, target, ignore_index=0, reduction="sum")
            a_l = a_l / mask.sum().item()
            a_loss += a_l.item()

            if reg is not None:
                t_l = F.mse_loss(
                    reg.view(-1)[mask], num.view(-1)[mask], reduction="sum"
                )
                t_l = t_l / mask.sum().item()
                t_loss += t_l.item()
                loss = a_l + t_l
            else:
                loss = a_l

            y_pred_class = torch.argmax(torch.softmax(logits, dim=1), dim=-1)
            a_acc += (
                y_pred_class[mask] == target[mask].view(-1)
            ).sum().item() / mask.sum().item()

        scaler.scale(loss).backward()
        # loss.backward()
        if grad_clip:
            clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

    a_loss /= len(data_loader)
    t_loss /= len(data_loader)
    a_acc /= len(data_loader)
    return a_loss, t_loss, a_acc
    # return a_loss, a_acc


def eval(model, data_loader):
    model.eval()
    a_loss, t_loss, a_acc = 0, 0, 0
    with torch.inference_mode():
        for items in data_loader:
            cat, num, constraints, target = (
                items["cat"],
                items["num"],
                items["constraints"],
                items["target"],
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                target = target.view(-1)
                mask = target != 0
                logits, reg, _ = model(x=(cat, num), constraints=constraints, mask=mask)

                logits = logits.view(-1, logits.shape[-1])

                target = target.view(-1)
                mask = target != 0

                a_l = F.cross_entropy(logits, target, ignore_index=0, reduction="sum")
                a_l = a_l / mask.sum().item()
                a_loss += a_l.item()

                if reg is not None:
                    t_l = F.mse_loss(
                        reg.view(-1)[mask], num.view(-1)[mask], reduction="sum"
                    )
                    t_l = t_l / mask.sum().item()
                    t_loss += t_l.item()

                y_pred_class = torch.argmax(torch.softmax(logits, dim=1), dim=-1)
                a_acc += (
                    y_pred_class[mask] == target[mask]
                ).sum().item() / mask.sum().item()

    a_loss /= len(data_loader)
    t_loss /= len(data_loader)
    a_acc /= len(data_loader)
    return a_loss, t_loss, a_acc
    # return a_loss, a_acc


def train(
    model: Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    # loss_fn: Union[Module, callable],
    optimizer: Optimizer,
    scaler,
    config: dict,
    scheduler=None,
):
    model.to(config["device"])
    results = {
        "train_a_loss": [],
        "train_t_loss": [],
        "train_a_acc": [],
        "test_a_loss": [],
        "test_t_loss": [],
        "test_a_acc": [],
    }
    best_acc = 0
    best_test_loss = float("inf")
    no_improve_epoch = 0
    for epoch in range(config["epochs"]):
        train_a_loss, train_t_loss, train_a_acc = train_step(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip=config["grad_clip"],
        )
        test_a_loss, test_t_loss, test_a_acc = eval(
            model=model,
            data_loader=test_loader,
        )
        scheduler.step()

        print(
            f"Epoch: {epoch+1} | "
            f"train_a_loss: {train_a_loss:.4f} | "
            f"test_a_loss: {test_a_loss:.4f} | "
            f"train_a_acc: {train_a_acc:.4f} | "
            f"test_a_acc: {test_a_acc:.4f} | "
            f"test_t_loss: {test_t_loss:.4f} |"
            f"train_t_loss: {train_t_loss:.4f} |"
        )

        # Update results dictionary
        results["train_a_loss"].append(train_a_loss)
        results["train_t_loss"].append(train_t_loss)
        results["train_a_acc"].append(train_a_acc)
        results["test_a_loss"].append(test_a_loss)
        results["test_t_loss"].append(test_t_loss)
        results["test_a_acc"].append(test_a_acc)

        if _has_wandb and config["wandb"]:
            wandb.log(
                {
                    "train_a_loss": train_a_loss,
                    "test_a_loss": test_a_loss,
                    "train_a_acc": train_a_acc,
                    "test_a_acc": test_a_acc,
                    "train_t_loss": train_t_loss,
                    "test_t_loss": test_t_loss,
                }
            )

        # check point
        # TODO: if checkpoint:
        cpkt = {
            "epoch": epoch,
            "test_a_acc": test_a_acc,
            "test_a_loss": test_a_loss,
            "test_t_loss": test_t_loss,
            "net": model.state_dict(),
            "optim": optimizer.state_dict(),
        }

        is_best = test_a_loss < best_test_loss
        if is_best:
            save_checkpoint(cpkt, config["run_name"], config)

        if test_a_loss < best_test_loss:
            best_test_loss = test_a_loss
            no_improve_epoch = 0
        else:
            no_improve_epoch += 1
        if no_improve_epoch > 30:
            break

    return results
