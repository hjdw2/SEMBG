import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import os, sys, random, shutil, argparse, time, logging, math, copy
import numpy as np
from ptflops import get_model_complexity_info

from data import *
from wrn import Wide_ResNet

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 training')
    parser.add_argument('--mode', choices=['CE'], default='CE')
    parser.add_argument('--data-dir', default='data', type=str,
                        help='the diretory to save cifar100 dataset')
    parser.add_argument('--arch', metavar='ARCH', default='resnet',
                        help='model architecture')
    parser.add_argument('--dataset', '-d', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4 )')
    parser.add_argument('--epoch', default=200, type=int,
                        help='number of total iterations (default: 64,000)')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual iter number (useful on restarts)')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='mini-batch size (default: 128)')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=5e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', default=100, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to  latest checkpoint (default: None)')
    parser.add_argument('--step_ratio', default=0.1, type=float,
                        help='ratio for learning rate deduction')
    parser.add_argument('--save-folder', default='save_checkpoints/', type=str,
                        help='folder to save the checkpoints')

    #distillation parameter
    parser.add_argument('--temperature', default=3, type=int, help='temperature to smooth the logits')
    parser.add_argument('--seed', default=1, type=int, help='random seed')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    save_path = args.save_path = os.path.join(args.save_folder, args.arch)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    args.logger_file = os.path.join(save_path, 'log.txt')
    handlers = [logging.FileHandler(args.logger_file, mode='w'), logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)

    logging.info('start training {}'.format(args.arch))
    run_training(args)

def run_training(args):
    best_prec = 0

    model = Wide_ResNet(28, 10, 0.0, 100)
    model = model.cuda()

    macs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    logging.info("=> Training Mode `{}`".format(args.mode))

    train_loader = prepare_cifar100_train_dataset(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.workers)
    test_loader = prepare_cifar100_test_dataset(data_dir=args.data_dir, batch_size=100, num_workers=args.workers)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(filter(lambda param: param.requires_grad, model.parameters()),
        lr=args.lr, momentum=args.momentum, weight_decay = args.weight_decay, nesterov=True)
    MSEloss = nn.MSELoss(reduction='mean').cuda()

    end = time.time()
    model.train()
    for current_epoch in range(args.start_epoch, args.epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        total_losses = AverageMeter()
        Acc = AverageMeter()

        adjust_learning_rate(args, optimizer, current_epoch)
        for i, (input, target) in enumerate(train_loader):
            data_time.update(time.time() - end)

            target = target.squeeze().long().cuda()
            input = Variable(input).cuda()
            output = model(input)

            CE = criterion(output, target)
            total_loss = CE

            total_losses.update(total_loss.item(), input.size(0))
            prec = accuracy(output.data, target, topk=(1,))
            Acc.update(prec[0], input.size(0))

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

        logging.info("Epoch: [{0}]\t"
                    "Iter: [{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                    "Prec@1 {Acc.val:.3f} ({Acc.avg:.3f})\t".format(
                        current_epoch,
                        i,
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=total_losses,
                        Acc=Acc)
        )

        prec = validate(args, test_loader, model, criterion)
        is_best = prec > best_prec
        best_prec = max(prec, best_prec)
        print("best: ", best_prec)

        if is_best:
            checkpoint_path = os.path.join(args.save_path, 'model_best.path.tar'.format(current_epoch))
            save_checkpoint({
                'state_dict': model.state_dict(),
                }, filename=checkpoint_path)
        torch.cuda.empty_cache()

    NLL_ECE(args, test_loader, model, checkpoint_path)

def validate(args, test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    Acc = AverageMeter()
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        target = target.squeeze().long().cuda()
        input = Variable(input).cuda()

        output = model(input)

        CE = criterion(output, target)
        losses.update(CE.item(), input.size(0))
        prec = accuracy(output.data, target, topk=(1,))
        Acc.update(prec[0], input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
    logging.info("Loss {loss.avg:.3f}\t"
                 "Main@1 {Acc.avg:.3f}\t".format(
                    loss=losses,
                    Acc=Acc))

    model.train()
    return Acc.avg

def kd_loss_function(output, target_output,args):
    output = output / args.temperature
    output_log_softmax = torch.log_softmax(output, dim=1)
    loss_kd = F.kl_div(output_log_softmax, target_output, reduction='sum')
    return loss_kd

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(args, optimizer, epoch):
    alpha = epoch / args.epoch
    if alpha <= 0.5:
        lr = args.lr
    elif alpha <= 0.9:
        lr = (1.0 - (alpha - 0.5) / 0.4 * 0.99) * args.lr
    else:
        lr = 0.01 * args.lr

    logging.info('Epoch [{}] learning rate = {}'.format(epoch, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul(100.0 / batch_size))
    return res

def save_checkpoint(state, filename):
    torch.save(state, filename)

def NLL_ECE(args, test_loader, model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    with torch.no_grad():
        model.eval()
        predictions = []
        for i, (input, _) in enumerate(test_loader):
            input = Variable(input).cuda()
            logits = []
            output = model(input)
            logits.append(output.cpu().numpy())
            predictions.append(logits)
        predictions = np.array(predictions, dtype=np.float).squeeze()
        predictions = torch.from_numpy(predictions)

        softmaxes = torch.softmax(predictions, dim=-1)
        softmaxes = torch.reshape(softmaxes, (len(test_loader.dataset), 100))
        predictions = torch.reshape(predictions, (len(test_loader.dataset), 100))

        labels = torch.tensor(test_loader.dataset.targets)

        _, predictions_acc = torch.max(predictions, -1)
        num_correct = (predictions_acc == labels).sum().data.item()
        print(f"ACC: {100 * num_correct / len(test_loader.dataset)}")
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(predictions, labels)
        print(f"NLL: {loss}")

        n_bins = 15
        boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = boundaries[:-1]
        bin_uppers = boundaries[1:]

        confidences, predictions = torch.max(softmaxes, -1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1).cuda()
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * \
                confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        print(f"ECE: {ece.item()}")

if __name__ == '__main__':
    main()
