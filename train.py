import argparse
import os
import numpy as np
import torch

from torch.utils.data import DataLoader
from dataset import MNistDataSet
from model import MNistNet
from tqdm import tqdm


def prepare_data_loaders(args, device):
    loader_config = {"batch_size": args.batch_size,
                     "shuffle": True,
                     "num_workers": args.num_workers}

    train_loader = DataLoader(MNistDataSet(args.train_data, device=device), **loader_config)
    valid_loader = DataLoader(MNistDataSet(args.valid_data, device=device), **loader_config)

    return train_loader, valid_loader


def print_config(args):
    print("\n[ CONFIGURATION ]\n")
    arg_len = max([len(arg) for arg, val in args.__dict__.items()]) + 1
    for arg, val in args.__dict__.items():
        print("{0:<{1:}} = {2:}".format(arg, arg_len, val))


def train_epoch(model, train_loader, optimizer, loss_fn):
    model.train()

    total_loss = 0
    epoch_accu = []

    for x, y in tqdm(train_loader, desc="[ Training ]", leave=False):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_correct = y.eq(y_pred.topk(1)[1].squeeze(-1)).sum().item()
        n_total = y.shape[0]

        epoch_accu.append((n_correct, n_total))

    n_correct_total = sum([e[0] for e in epoch_accu])
    n_total = sum([e[1] for e in epoch_accu])

    total_loss = total_loss / n_total

    return total_loss, n_correct_total / n_total


def eval_epoch(model, valid_loader, loss_fn):
    model.eval()

    total_loss = 0
    epoch_accu = []

    with torch.no_grad():
        for x, y in tqdm(valid_loader, desc="[ Validation ]", leave=False):
            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            total_loss += loss.item()
            n_correct = y.eq(y_pred.topk(1)[1].squeeze(-1)).sum().item()
            n_total = y.shape[0]

            epoch_accu.append((n_correct, n_total))

    n_correct_total = sum([e[0] for e in epoch_accu])
    n_total = sum([e[1] for e in epoch_accu])

    total_loss = total_loss / n_total

    return total_loss, n_correct_total / n_total


def train(args, model, train_loader, valid_loader):
    hist_loss_train = []
    hist_loss_val = []
    hist_accu_train = []
    hist_accu_val = []

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    if args.log_file:
        with open(args.log_file, "w") as f:
            f.write('epoch,loss_train,accu_train,loss_val,accu_val\n')

    for epoch in range(args.epochs):
        print("[ Epoch {} / {} ]".format(epoch, args.epochs))

        loss_train, accu_train = train_epoch(model, train_loader, optimizer, loss_fn)
        hist_loss_train.append(loss_train)
        hist_accu_train.append(accu_train)

        loss_val, accu_val = eval_epoch(model, valid_loader, loss_fn)
        hist_loss_val.append(loss_val)
        hist_accu_val.append(accu_val)

        print("[ METRIC ] Epoch {epoch:}: loss_train = {loss_train: 8.5f}, loss_val = {loss_val: 8.5f}, "
              "accu_train = {accu_train: 3.3f}, accu_val = {accu_val: 3.3f}".format(
                  epoch=epoch, loss_train=loss_train, loss_val=loss_val, accu_train=accu_train, accu_val=accu_val))

        if args.log_file:
            vals = [epoch, loss_train, accu_train, loss_val, accu_val]
            with open(args.log_file, "a") as f:
                f.write(",".join(["{:8.5f}".format(e) for e in vals]) + "\n")

        if args.model_dir:
            if accu_val >= max(hist_accu_val):
                filename = os.path.join(args.model_dir, "model.pt")
                torch.save(model.state_dict(), filename)
                if args.verbose:
                    print("[INFO] Best model saved")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train-data', type=str, required=True)
    parser.add_argument('--valid-data', type=str, required=True)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--no-cuda', dest="no_cuda", action='store_true')
    parser.add_argument('--verbose', dest="verbose", action="store_true")

    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--model-dir', type=str)
    parser.add_argument('--log-file', type=str)

    args = parser.parse_args()
    args.cuda = not args.no_cuda
    args.device = 'cuda' if args.cuda else 'cpu'
    device = torch.device(args.device)

    if args.verbose:
        print_config(args)

    if args.verbose:
        print("\n### Creating Data Loaders ###\n")

    model = MNistNet(args.dropout).to(device)

    train_loader, valid_loader = prepare_data_loaders(args, device=device)

    if args.verbose:
        params = sum([np.prod(vals.size()) for _, vals in model.state_dict().items()])
        print("Model has {:,} trainable parameters\n".format(params))

    train(args, model, train_loader, valid_loader)


if __name__ == "__main__":
    main()
