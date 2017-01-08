import argparse
import pickle 
import matplotlib.pyplot as plt
import numpy as np
import chainer.links
import chainer.functions as F
import chainer.links as L
import os

from chainer import training


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


class visualize_param():
    def __init__(self, model):
        self.model = model

    def visualize_all(self, args):
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)

        self.params = self.__getParam()
        j = 0
        print(self.params)
        for param in self.params:
            self.__visualize(self.model.__dict__[param], args, param)
            print("j=%d\n" % j)
            j += 1

    def __getParam(self):
        params = []
        i = 0        
        for key in self.model.__dict__.keys():
            print("i=%d" % i)
            i+=1
            print(key)
            if isinstance(self.model.__dict__[key], chainer.links.connection.linear.Linear):
                params.append(key)
                i+=1
        return params

    def __visualize(self, filter, args, param):
        outname = args.outdir + "/" + param
        weights = filter.W
        out_n, in_n, h, w = weights.shape

        if not args.hshape:
            out_n = args.hshape

        if not args.vshape:
            in_n = args.vshape

        fig = plt.figure()
        fig.subplots_adjust(left=args.left, right=args.right, bottom=args.bottom, top=args.top, hspace=args.hspace, wspace=args.wspace)
        self.weights = []

        for i in range(out_n):
            for j in range(in_n):
                weight = weights[i, j].data
                self.weights.append(weight)
                ax = fig.add_subplot(out_n, in_n, in_n * i + j + 1, xticks=[], yticks=[])
                ax.imshow(weight, cmap=plt.cm.gray_r, interpolation='nearest')

        self.weights = np.asarray(self.weights, np.float)
        plt.savefig(outname + ".png")


# Main Function
def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--left', '-l', type=int, default=0,
                        help='Left Pitch [pixels]')
    parser.add_argument('--right', '-r', type=int, default=1,
                        help='Right Pitch [pixels]')
    parser.add_argument('--bottom', '-b', type=int, default=0,
                        help='Bottom Pitch [pixels]')
    parser.add_argument('--hspace', type=float, default=0.05,
                        help='Horizontal Space')
    parser.add_argument('--wspace', type=float, default=0.05,
                        help='Vertical Space')
    parser.add_argument('--hshape', '-hs', type=float,
                        help='Horizontal Shape')
    parser.add_argument('--vshape', '-vs', type=float,
                        help='Vertical Shape')
    parser.add_argument('--outdir', '-o', default='./',
                        help='Output Path')
    parser.add_argument('--out_name', '-n', default='param',
                        help='Output File Name')
    parser.add_argument('--model_path', '-m', default='./my.model',
                        help='Model File Name and Path')
    args = parser.parse_args()

    # Set up a neural network
    model = L.Classifier(MLP(args.unit, 10))
    model.train = False

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()
    train_iter = chainer.iterators.SerialIterator(train, 100)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=-1)
    trainer = training.Trainer(updater, (20, 'epoch'), out="v_result")

    # load model
    model = training.Trainer(updater, (20, 'epoch'), out="v_result")

    chainer.serializers.load_npz('./my.model', model)
    classifier_model = trainer.updater.get_optimizer('main').target
    mlp_model = classifier_model.predictor

    # Make Visualizer
    vp = visualize_param(classifier_model)

    # Visualize
    vp.visualize_all(args)
    
if __name__ == '__main__':
    main()