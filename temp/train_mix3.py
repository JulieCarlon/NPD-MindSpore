#!/usr/bin/env python
# coding: utf-8

from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.communication.management import init
from mindspore.nn.optim.momentum import Momentum
from mindspore.nn.optim import Adam
from mindspore.nn.optim import SGD
from mindspore import context
from models.resnet import resnet18
import numpy as np

from attack.utils import *

def main(args):
    set_seed(args.seed)
    # Step 1: Load dataset
    # poison train data
    trans_train, trans_label_train = get_trans(args.data, args.image_size, train=True)
    trans_test, trans_label_test = get_trans(args.data, args.image_size, train=False)

    # Load the training and testing datasets.
    data_bd_train = np.load(f'./poison_data/{args.data}/{args.poison_data}/train_bd.npz')
    data_bd_test = np.load(f'./poison_data/{args.data}/{args.poison_data}/test_bd.npz')
    data_clean_train = np.load(f'./poison_data/{args.data}/{args.poison_data}/train_clean.npz')
    data_clean_test = np.load(f'./poison_data/{args.data}/{args.poison_data}/test_clean.npz')

    # Create the dataset iterators for the train and test sets    
    iter_bd_train = poison_iterator(data = data_bd_train['data'], label = data_bd_train['labels'])
    iter_bd_test = poison_iterator(data = data_bd_test['data'], label = data_bd_test['labels'])
    iter_clean_train = poison_iterator(data = data_clean_train['data'], label = data_clean_train['labels'])
    iter_clean_test = poison_iterator(data = data_clean_test['data'], label = data_clean_test['labels'])

    # Create the dataset objects
    dataset_bd_train = GeneratorDataset(source=iter_bd_train, column_names=["data", "label"])
    dataset_bd_test = GeneratorDataset(source=iter_bd_test, column_names=["data", "label"])
    dataset_clean_train = GeneratorDataset(source=iter_clean_train, column_names=["data", "label"])
    dataset_clean_test = GeneratorDataset(source=iter_clean_test, column_names=["data", "label"])

    # Add transform operation
    dataset_bd_train = dataset_bd_train.map(operations=trans_train, input_columns='data')
    dataset_bd_test = dataset_bd_test.map(operations=trans_test, input_columns='data')
    dataset_clean_train = dataset_clean_train.map(operations=trans_train, input_columns='data')
    dataset_clean_test = dataset_clean_test.map(operations=trans_test, input_columns='data')

    # Add transform to label
    dataset_bd_train = dataset_bd_train.map(operations=trans_label_train, input_columns='label')
    dataset_bd_test = dataset_bd_test.map(operations=trans_label_test, input_columns='label')
    dataset_clean_train = dataset_clean_train.map(operations=trans_label_train, input_columns='label')
    dataset_clean_test = dataset_clean_test.map(operations=trans_label_test, input_columns='label')

    # Shuffle
    dataset_bd_train = dataset_bd_train.shuffle(buffer_size=10)
    dataset_clean_train = dataset_clean_train.shuffle(buffer_size=10)


    # Add batch dimension
    dataset_bd_train = dataset_bd_train.batch(batch_size=args.batch_size, drop_remainder=True)
    dataset_bd_test = dataset_bd_test.batch(batch_size=args.batch_size, drop_remainder=False)
    dataset_clean_train = dataset_clean_train.batch(batch_size=args.batch_size, drop_remainder=True)
    dataset_clean_test = dataset_clean_test.batch(batch_size=args.batch_size, drop_remainder=False)


    context.set_context(mode=1, device_target="GPU")
    num_classes = args.num_classes

    net = resnet18(num_classes)
    ls = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    if args.opt == 'Momentum':
        opt = Momentum(net.trainable_params(),args.lr, args.momentum)
    if args.opt == 'Adam':
        opt = Adam(net.trainable_params(),args.lr, args.weight_decay)
    if args.opt == 'SGD':
        opt = SGD(net.trainable_params(),args.lr, args.weight_decay)

    train_data = dataset_clean_train
    eval_data = dataset_clean_test
    eval_data_bd = dataset_bd_test
    # Define forward function
    def forward_fn(data, label):
        logits = net(data)
        loss = ls(logits, label)
        return loss, logits

    # Define grad function
    grad_fn = ops.value_and_grad(forward_fn, None, opt.parameters, has_aux=True)


    def test(net, dataset, loss_fn):
        num_batches = dataset.get_dataset_size()
        net.set_train(False)
        total, test_loss, correct = 0, 0, 0
        for data, label in dataset.create_tuple_iterator():
            pred = net(data)
            total += len(data)
            test_loss += loss_fn(pred, label).asnumpy()
            correct += (pred.argmax(1) == label).asnumpy().sum()
        test_loss /= num_batches
        correct /= total
        # print(f"Test: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return correct

    # Step 3: Training
    for epoch in range(args.num_epoch):
        net.set_train()
        for data, label in train_data.create_tuple_iterator():
            (loss, _), grads = grad_fn(data, label)
            loss = ops.depend(loss, opt(grads))

        print(f"Epoch [{epoch+1}], loss: {loss.asnumpy():.4f}")
        acc = test(net, eval_data, ls)
        print(f"Clean test accuracy: {acc}")
        asr = test(net, eval_data_bd, ls)
        print(f"Backdoor test accuracy (ASR): {asr}")

    # Step 4: Save model
    ms.save_checkpoint(net, f"record/{args.data}_{args.model}_{args.poison_data}model.ckpt")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--opt", type=str, default="Adam")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--poison_data", type=str, default="badnets10")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.data == "cifar10":
        args.num_samples = 50000
        args.num_classes = 10
    elif args.data == "tiny-imagenet":
        args.num_samples = 100000
        args.num_classes = 200
    elif args.data == "imagenet":
        args.num_samples = 1281167
        args.num_classes = 1000
    elif args.data == "cifar100":
        args.num_samples = 50000
        args.num_classes = 100
    elif args.data == "gstrb":
        args.num_samples = 39209
        args.num_classes = 43

    main(args)    