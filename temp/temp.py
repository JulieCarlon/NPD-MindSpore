"""train resnet."""
import numpy as np
import mindspore as ms
from mindspore import ops
import mindspore.nn as nn
from mindspore import nn
from mindspore.nn.optim import Momentum, thor, LARS, AdamWeightDecay, Adam
from mindspore.common import set_seed
from mindspore.dataset import GeneratorDataset
from mindspore import dtype as mstype
from models.resnetv2 import resnet18
from argparse import ArgumentParser
from attack.utils import *


def test(model, dataset, loss_fn):
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for data, label in dataset.create_tuple_iterator():
        pred = model(data)
        total += len(data)
        test_loss += loss_fn(pred, label).asnumpy()
        correct += (pred.argmax(1) == label).asnumpy().sum()
    test_loss /= num_batches
    correct /= total
    print(f"Test: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct




def main(args):
    set_seed(args.seed)
    ms.set_context(mode=args.mode)
    # Step 1: Load dataset
    # poison train data
    trans_train, _ = get_trans(args.data, args.image_size, train=True)
    trans_test, _ = get_trans(args.data, args.image_size, train=False)

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
    
    # Add batch dimension
    dataset_bd_train = dataset_bd_train.batch(batch_size=args.batch_size, drop_remainder=False)
    dataset_bd_test = dataset_bd_test.batch(batch_size=args.batch_size, drop_remainder=False)
    dataset_clean_train = dataset_clean_train.batch(batch_size=args.batch_size, drop_remainder=False)
    dataset_clean_test = dataset_clean_test.batch(batch_size=args.batch_size, drop_remainder=False)

    
    # Step 2: Load model & set loss, opt, forward_fn, grad_fn
    if args.model == 'resnet18':
        model = resnet18(class_num = args.num_classes)
    loss_fn = nn.CrossEntropyLoss()
    opt = Adam(params=model.trainable_params(), learning_rate=0.01, weight_decay=1e-4)
    # opt = Momentum(params=model.trainable_params(), learning_rate=0.01, momentum=0.9, weight_decay=1e-4)
    
    # Define forward function
    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss, logits
    
    # Define grad function
    grad_fn = ops.value_and_grad(forward_fn, None, opt.parameters, has_aux=True)
    from mindspore import load_checkpoint
    load_checkpoint("/workspace2/weishaokui/Resnet-18/checkpoint_40eps.ckpt", net=model)


    train_acc = test(model, dataset_bd_train, loss_fn)
    print(f"Train asr: {train_acc}")

    train_acc = test(model, dataset_clean_train, loss_fn)
    print(f"Train acc: {train_acc}")
    
    print('Test on clean')
    acc = test(model, dataset_clean_test, loss_fn)
    
    print('Test on BD')
    asr = test(model, dataset_bd_test, loss_fn)
    print(f'Acc: {acc}\nASR: {asr}')
        
    # Step 4: Save model
    ms.save_checkpoint(model, f"record/{args.data}_{args.model}_{args.poison_data}model.ckpt")




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--mode", type=int, default=0)
    parser.add_argument("--poison_data", type=str, default="badnets10")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
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
    