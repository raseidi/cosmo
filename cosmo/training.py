import os
import torch
from .utils import save_checkpoint, ensure_dir


def train_step(
    model,
    data_loader,
    criterion_clf,
    criterion_res,
    criterion_rt,
    device,
    optimizer,
    sc=None,
):
    model.train()
    loss_rt, loss_res, loss_clf, correct = 0, 0, 0, 0
    for X, y in data_loader:
        x = [d.to(device) for d in X]
        y = [d.to(device) for d in y]

        na, res, rt, _ = model(x)

        l_clf = criterion_clf(na, y[0].long())
        if "resource" in model.vocabs.keys():
            # ToDo: manage event attributes in a better/more generic/scalable manner
            l_res = criterion_res(res, y[1].long())
        else:
            l_res = criterion_res(res, y[1].unsqueeze(1))
        l_rt = criterion_rt(rt, y[2].unsqueeze(1))

        loss_clf += l_clf.item() * len(X[0])  # reduction=mean
        loss_res += l_res.item() * len(X[0])  # reduction=mean
        loss_rt += l_rt.item() * len(X[0])

        loss = l_clf + l_res + l_rt
        loss.backward()
        optimizer.step()
        # see https://stackoverflow.com/a/46820512
        model.zero_grad()  # this can be outside the loop

        y_pred_class = torch.argmax(torch.softmax(na, dim=1), dim=1)
        correct += (y_pred_class == y[0]).sum().item()
    if sc:
        sc.step()
    loss_clf /= len(data_loader.dataset)
    loss_rt /= len(data_loader.dataset)  # test loss is what matters tho
    loss_res /= len(data_loader.dataset)
    acc = correct / len(data_loader.dataset)
    return loss_clf, acc, loss_rt, loss_res


def eval(model, data_loader, criterion_clf, criterion_res, criterion_rt, device):
    model.eval()
    loss_clf, loss_rt, loss_res, correct = 0, 0, 0, 0
    with torch.inference_mode():
        for X, y in data_loader:
            x = [d.to(device) for d in X]
            y = [d.to(device) for d in y]

            na, res, rt, _ = model(x)

            # todo: mask pad tokens
            # see https://discuss.pytorch.org/t/ignore-padding-area-in-loss-computation/95804/5
            # no need for masking since we are doing left padding
            l_clf = criterion_clf(na, y[0].long())
            if "resource" in model.vocabs.keys():
                l_res = criterion_res(res, y[1].long())
            else:
                l_res = criterion_res(res, y[1].unsqueeze(1))
            l_rt = criterion_rt(rt, y[2].unsqueeze(1))

            loss_clf += l_clf.item() * len(X[0])  # reduction=mean
            loss_rt += l_rt.item() * len(X[0])
            loss_res += l_res.item() * len(X[0])

            y_pred_class = torch.argmax(torch.softmax(na, dim=1), dim=1)
            correct += (y_pred_class == y[0]).sum().item()

    loss_clf /= len(data_loader.dataset)
    loss_rt /= len(data_loader.dataset)
    loss_res /= len(data_loader.dataset)
    acc = correct / len(data_loader.dataset)
    return loss_clf, acc, loss_rt, loss_res


def train(
    model,
    train_loader,
    test_loader,
    loss_fn,
    optimizer,
    sc=None,
    logger=None,
    check_point=None,
):
    # ToDo: handle when logger is None
    model_path = (
        f"models/{logger.config.dataset}/{logger.config.condition}/{logger.name}/"
    )
    ensure_dir(model_path)
    model.to(logger.config.device)
    best_acc = 0
    for epoch in range(logger.config.epochs):
        train_loss, train_acc, train_loss_rt, train_loss_res = train_step(
            model=model,
            data_loader=train_loader,
            criterion_clf=loss_fn["clf"],
            criterion_res=loss_fn["res"],
            criterion_rt=loss_fn["reg"],
            device=logger.config.device,
            optimizer=optimizer,
            sc=sc,
        )
        test_loss, test_acc, test_loss_rt, test_loss_res = eval(
            model=model,
            data_loader=test_loader,
            criterion_clf=loss_fn["clf"],
            criterion_res=loss_fn["res"],
            criterion_rt=loss_fn["reg"],
            device=logger.config.device,
        )

        # logging time error in days
        test_loss_rt = torch.exp(torch.tensor(test_loss_rt)).item() / (24 * 60 * 60)
        train_loss_rt = torch.exp(torch.tensor(train_loss_rt)).item() / (24 * 60 * 60)
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_acc: {test_acc:.4f} | "
            f"train_loss_rt: {train_loss_rt:.4f} days| "  # outcome is in log secs
            f"test_loss_rt: {test_loss_rt:.4f} days| ",
            f"train_loss_res: {train_loss_res:.4f}",
            f"test_loss_res: {test_loss_res:.4f}",
        )

        if logger:
            logger.log(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "train_loss_rt(days)": train_loss_rt,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "test_loss_rt(days)": test_loss_rt,
                    "train_loss_res": train_loss_res,
                    "test_loss_res": test_loss_res,
                }
            )
        is_best = test_acc > best_acc
        best_acc = test_acc if is_best else best_acc

        cpkt = {
            "net": model.state_dict(),
            "epoch": epoch,
            "test_acc": test_acc,
            "optim": optimizer.state_dict(),
        }
        save_checkpoint(
            cpkt, os.path.join(model_path, "checkpoint.ckpt"), is_best=is_best
        )

    return
