#!/usr/bin/env python
from __future__ import print_function
import argparse

import numpy as np
import matplotlib.pyplot as plt

import chainer
import chainer.functions as F
import chainer.links as L

from chainer import optimizers
from chainer import training
from chainer import cuda


# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(None, n_units),  # n_in -> n_units
            l2=L.Linear(None, n_units),  # n_units -> n_units
            l3=L.Linear(None, n_out),  # n_units -> n_out
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


# Main Function
def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--num_inf', '-i', type=int, default=15,
                        help='Number of Inferences')
    parser.add_argument('--fig_num_clm', '-c', type=int, default=5,
                        help='Number of Plotting Columns in Figure')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('')

    # Set up a neural network
    model = MLP(args.unit, 10)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU
    
    xp = np if args.gpu < 0 else cuda.cupy
 
    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()

    # load model    
    chainer.serializers.load_npz('./my.model', model)
    optimizer = optimizers.Adam(alpha=0.00005)
    optimizer.setup(model)

    # show graphical results of first 15 data to understand what's going on in inference stage
    plt.figure(figsize=(args.num_inf, 10))

    cnt_ok = 0
    for i in range(args.num_inf):
        # Input vector
        x = chainer.Variable(xp.asarray([test[i][0]]))  # test data
        
        # Forwward Propagation
        y = model(x)
        F.softmax(y).data
        
        # Predication
        prediction = y.data.argmax(axis=1)
        predict_awnswer = max(prediction)
        
        # Decision of OK/NG
        decision = "NG"
        if predict_awnswer == test[i][1]:
            cnt_ok += 1
            decision = "OK"

        print('%04d-th image: answer = %d, predict = %d: %s' % (i, test[i][1], predict_awnswer, decision))
        
        np.set_printoptions(precision=2, suppress=True)
        test_image = (test[i][0] * 255).astype(np.int32).reshape(28, 28)
        plt.subplot(args.num_inf/args.fig_num_clm, args.fig_num_clm, i+1)
        plt.imshow(test_image, cmap='gray')
        plt.title("No.{0} / Answer:{1}, Predict:{2}".format(i, test[i][1], prediction))
        plt.axis("off")

#    plt.tight_layout()
    plt.savefig('{}/predict.png'.format(args.out))

if __name__ == '__main__':
    main()
