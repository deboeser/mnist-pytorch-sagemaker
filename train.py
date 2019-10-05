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
    hist_train_loss = []
    hist_val_loss = []
    hist_train_accu = []
    hist_val_accu = []

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    if args.output_data_dir:
        with open(args.log_file, "w") as f:
            f.write('epoch,train_loss,train_accu,val_loss,val_accu\n')

    for epoch in range(args.epochs):
        print("[ Epoch {} / {} ]".format(epoch, args.epochs))

        train_loss, train_accu = train_epoch(model, train_loader, optimizer, loss_fn)
        hist_train_loss.append(train_loss)
        hist_train_accu.append(train_accu)

        val_loss, val_accu = eval_epoch(model, valid_loader, loss_fn)
        hist_val_loss.append(val_loss)
        hist_val_accu.append(val_accu)

        print("[ METRIC ] Epoch {epoch:}: train_loss = {train_loss:8.5f}, val_loss = {val_loss:8.5f}, "
              "train_accu = {train_accu:3.3f}, val_accu = {val_accu:3.3f}".format(
                  epoch=epoch, train_loss=train_loss, val_loss=val_loss, train_accu=train_accu, val_accu=val_accu))

        if args.sagemaker_logging:
            print("train_loss={train_loss:8.8f};  val_loss={val_loss:8.8f};  "
                  "train_accu={train_accu:1.5f};  val_accu={val_accu:1.5f};".format(
                      train_loss=train_loss, val_loss=val_loss, train_accu=train_accu, val_accu=val_accu))

        if args.output_data_dir:
            vals = [epoch+1, train_loss, train_accu, val_loss, val_accu]
            with open(args.log_file, "a") as f:
                f.write(",".join(["{:8.5f}".format(e) for e in vals]) + "\n")

        if args.model_dir:
            if val_accu >= max(hist_val_accu):
                filename = os.path.join(args.model_dir, "model.pt")
                torch.save(model.state_dict(), filename)
                if args.verbose:
                    print("[INFO] Best model saved")


def main():
    parser = argparse.ArgumentParser()

    if 'SM_MODEL_DIR' in os.environ:
        parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
        parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
        parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    else:
        parser.add_argument('--model-dir', type=str, default=None)
        parser.add_argument('--output-data-dir', type=str, default=None)
        parser.add_argument('--data-dir', type=str, required=True)

    parser.add_argument('--train-data-file', type=str, required=True)
    parser.add_argument('--valid-data-file', type=str, required=True)
    parser.add_argument('--model-name', type=str, default="model")

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--no-cuda', dest="no_cuda", action='store_true')
    parser.add_argument('--verbose', dest="verbose", action="store_true")

    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--sagemaker-logging', dest="sagemaker_logging", action="store_true")

    args = parser.parse_args()
    args.cuda = not args.no_cuda
    args.device = 'cuda' if args.cuda else 'cpu'
    device = torch.device(args.device)

    if args.verbose:
        print_config(args)

    args.train_data = os.path.join(args.data_dir, args.train_data_file)
    args.valid_data = os.path.join(args.data_dir, args.valid_data_file)

    if args.output_data_dir:
        args.log_file = os.path.join(args.output_data_dir, args.model_name + "_log.csv")

    if args.model_dir:
        args.model_file = os.path.join(args.model_dir, args.model_name + "_model.pt")

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
