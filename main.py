# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

import data
import models

parser = argparse.ArgumentParser(description='RNNs with Auxiliary Losses')
parser.add_argument('--dataset', type=str, default='MNIST',
                    help='type of dataset (MNIST/pMNIST/CIFAR)')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=128,
                    help='size of embeddings for auxiliary network')
parser.add_argument('--nhid', type=int, default=128,
                    help='number of hidden units per layer')
parser.add_argument('--nhid_ffn', type=int, default=256,
                    help='number of hidden units in FFN')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers for main network')
parser.add_argument('--aux_nlayers', type=int, default=2,
                    help='number of layers for auxiliary network')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--pre_epochs', type=int, default=100,
                    help='upper epoch limit for pretraining')
parser.add_argument('--epochs', type=int, default=1000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=300,
                    help='truncated bptt length')
parser.add_argument('--aux_length', type=int, default=600,
                    help='subsequence length for auxiliary network')
parser.add_argument('--aux_frequency', type=int, default=1,
                    help='sampling frequency for auxiliary network')
parser.add_argument('--scheduled_sampling', action='store_true',
                    help='train auxiliary network with scheduled sampling')
parser.add_argument('--dropconnect', type=float, default=0.5,
                    help='dropconnect applied to ffn in auxiliary network (0 = no dropconnect)')
parser.add_argument('--seed', type=int, default=1004,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load tensorboardX writer
###############################################################################

writer = SummaryWriter()

###############################################################################
# Load data
###############################################################################

if args.dataset in ['MNIST', 'pMNIST']:
    permute = True if args.dataset == 'pMNIST' else False
    train_loader, valid_loader, test_loader = data.sequential_mnist(
        args.batch_size, permute)
elif args.dataset == 'CIFAR':
    train_loader, valid_loader, test_loader = data.sequential_cifar10(
        args.batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = 255
model = models.MainRNNModel(args.model, 1, args.nhid, args.nhid_ffn, args.nlayers,
                            args.dropconnect).to(device)
aux_model = models.AuxiliaryRNNModel(args.model, ntokens, args.emsize, args.nhid, args.nhid_ffn,
                                     args.aux_nlayers, args.dropconnect).to(device)

optimizer = torch.optim.RMSprop(list(model.parameters()) + list(aux_model.parameters()), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)

if args.pre_epochs != 0:
    pre_optimizer = torch.optim.RMSprop(list(model.parameters()) + list(aux_model.parameters()), lr=args.lr)
    pre_scheduler = torch.optim.lr_scheduler.StepLR(pre_optimizer, step_size=1,
                                                    gamma=(0.5)**(1/args.pre_epochs))

criterion = nn.CrossEntropyLoss()
anchor = None

#########################s######################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
    length = source.size()[0]
    if i != 0 or args.bptt == length:
        seq_len = args.bptt
    else:
        seq_len = length % args.bptt
    data = source[i:i+seq_len]
    return data


def get_aux_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def cut_sequence(length, bptt):
    indices = [length - i * bptt for i in range(length // bptt, 0, -1)]
    if 0 not in indices:
        indices = [0] + indices
    return indices


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    total = 0.
    correct = 0.
    with torch.no_grad():
        for image, target in data_source:
            image = image.t().to(device)
            target = target.to(device)
            hidden = model.init_hidden(args.batch_size)
            for i in cut_sequence(image.size(0), args.bptt):
                hidden = repackage_hidden(hidden)
                data = get_batch(image, i)
                hidden = model(data, hidden)
            output = model.out(hidden)
            predicted = torch.argmax(output, dim=1)

            total_loss += criterion(output, target).item()
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return total_loss / len(data_source), correct / total


def pretrain():
    # Turn on training mode which enables dropout.
    model.train()
    aux_model.train()
    total_aux_loss = 0.
    start_time = time.time()
    for batch_idx, (image, _) in enumerate(train_loader):
        image = image.t().to(device)
        hidden = model.init_hidden(args.batch_size)

        # Select random anchor point
        global anchor
        if not anchor:
            anchor = torch.randint(args.aux_length, image.size(0), (1,1)).long().item()

        # Get hidden state at anchor point
        for i in cut_sequence(anchor + 1, args.bptt):
            hidden = repackage_hidden(hidden)
            data = get_batch(image[:anchor + 1], i)
            hidden = model(data, hidden)
        
        aux_hidden = aux_model.init_hidden(args.batch_size, hidden)

        subsequence = image[anchor - args.aux_length : anchor]

        for j in cut_sequence(args.aux_length, args.bptt):
            aux_data, aux_target = get_aux_batch(subsequence, j)
            model.zero_grad()
            aux_model.zero_grad()
            aux_output, aux_hidden = aux_model(aux_data, aux_hidden)

            aux_loss = criterion(aux_output.view(-1, ntokens), aux_target)
            aux_loss.backward()
            pre_optimizer.step()
            aux_hidden = repackage_hidden(aux_hidden)
            
            total_aux_loss += (aux_loss.item() / len(cut_sequence(args.aux_length, args.bptt)))
            
        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            cur_aux_loss = total_aux_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:04.4f} | ms/batch {:5.2f} | '
                'loss {:5.2f} | ppl {:8.2f}'.format(
            epoch, batch_idx, len(train_loader), pre_scheduler.get_lr()[0],
            elapsed * 1000 / args.log_interval, cur_aux_loss, math.exp(cur_aux_loss)))
            total_aux_loss = 0
            start_time = time.time()


def train():
    # Turn on training mode which enables dropout.
    model.train()
    aux_model.train()
    total_loss = total_aux_loss = 0.
    total = correct = 0.

    start_time = time.time()
    for batch_idx, (image, target) in enumerate(train_loader):
        image = image.t().to(device)
        target = target.to(device)

        # Select random anchor point if not selected in pretraining step
        global anchor
        if not anchor:
            anchor = torch.randint(args.aux_length, image.size(0), (1,1)).long().item()

        # Does the graph of auxiliary network and main network overlap?
        retain_graph = anchor > (image.size(0) - args.bptt - 1)

        init_hidden = model.init_hidden(args.batch_size)
        hiddens = [(None, init_hidden)]
        model.zero_grad()
        aux_model.zero_grad()

        for i in range(image.size(0)):
            hidden = repackage_hidden(hiddens[-1][1])
            hidden[0].requires_grad = True
            hidden[1].requires_grad = True
            new_hidden = model(image[i:i+1], hidden)
            hiddens.append((hidden, new_hidden))
            
            while len(hiddens) > (image.size(0) - (anchor - args.aux_length)):
                # Delete stuff that is too old
                del hiddens[0]

            if i == anchor:
                aux_hidden = aux_model.init_hidden(args.batch_size, new_hidden)
                subsequence = image[anchor - args.aux_length : anchor]

                for j in cut_sequence(args.aux_length, args.bptt):
                    aux_data, aux_target = get_aux_batch(subsequence, j)
                    aux_output, aux_hidden = aux_model(aux_data, aux_hidden)

                    aux_loss = criterion(aux_output.view(-1, ntokens), aux_target)
                    aux_loss.backward(retain_graph=retain_graph)
                    
                    # Backprop for bptt timesteps in main network
                    if j == 0:
                        for k in range(args.bptt - 1):
                            # if we get all the way back to the "init_hidden", stop
                            if hiddens[-k-2][0] is None:
                                break
                            curr_h_grad = hiddens[-k-1][0][0].grad
                            curr_c_grad = hiddens[-k-1][0][1].grad
                            hiddens[-k-2][1][0].backward(curr_h_grad, retain_graph=retain_graph)
                            hiddens[-k-2][1][1].backward(curr_c_grad, retain_graph=retain_graph)
                    
                    aux_hidden = repackage_hidden(aux_hidden)
                    total_aux_loss += (aux_loss.item() / len(cut_sequence(args.aux_length, args.bptt)))

        output = model.out(new_hidden)
        predicted = torch.argmax(output, dim=1)

        loss = criterion(output, target)
        loss.backward()

        for k in range(args.bptt - 1):
            if hiddens[-k-2][0] is None:
                break
            curr_h_grad = hiddens[-k-1][0][0].grad
            curr_c_grad = hiddens[-k-1][0][1].grad
            hiddens[-k-2][1][0].backward(curr_h_grad, retain_graph=retain_graph)
            hiddens[-k-2][1][1].backward(curr_c_grad, retain_graph=retain_graph)
        
        optimizer.step()

        total_loss += loss.item()
        total += target.size(0)
        correct += (predicted == target).sum().item()

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            cur_loss = total_loss / args.log_interval
            cur_aux_loss = total_aux_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:04.4f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | aux_loss {:5.2f} | accuracy {:8.2f}'.format(
                epoch, batch_idx, len(train_loader), scheduler.get_lr()[0],
                elapsed * 1000 / args.log_interval, cur_loss, cur_aux_loss, (correct / total)))

            # Log scalars to tensorboard
            n_iter = (epoch - 1) * len(train_loader) + batch_idx
            writer.add_scalars('data/loss', {'train': cur_loss}, n_iter)
            writer.add_scalars('data/accuracy', {'train': correct/total}, n_iter)
            writer.add_scalar('data/lr', scheduler.get_lr()[0], n_iter)
            writer.add_scalar('data/aux_loss', cur_aux_loss, n_iter)

            total_loss = 0.
            total_aux_loss = 0.
            total = 0.
            correct = 0.
            start_time = time.time()

# Loop over epochs.
best_val_loss = None
best_val_acc = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    print('-' * 89)
    print('| Pre-Training for {:3d} Epochs |'.format(args.pre_epochs))
    print('-' * 89)
    for epoch in range(1, args.pre_epochs+1):
        epoch_start_time = time.time()
        pre_scheduler.step()
        pretrain()
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s |'.format(epoch, (time.time() - epoch_start_time)))
        print('-' * 89)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from pre-training early')

try:
    print('-' * 89)
    print('| Training for {:3d} Epochs |'.format(args.epochs))
    print('-' * 89)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        scheduler.step()
        train()
        val_loss, val_acc = evaluate(valid_loader)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid accuracy {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, val_acc))
        print('-' * 89)

        # Log scalars to tensorboard
        n_iter = epoch * len(train_loader)
        writer.add_scalars('data/loss', {'valid': val_loss}, n_iter)
        writer.add_scalars('data/accuracy', {'valid': val_acc}, n_iter)

        # Save the model if the validation accuracy is the best we've seen so far.
        if not best_val_acc or val_acc > best_val_acc:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_acc = val_acc

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()

# Run on test data.
test_loss, test_acc = evaluate(test_loader)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test accuracy {:8.2f}'.format(
    test_loss, test_acc))
print('=' * 89)