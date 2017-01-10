import argparse
import chainer.functions as F
import chainer.links as L
import chainer.links

# Pixel Configuration: 28*28=784
img_width = 28
img_height = 28
length = img_width * img_height


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


class statics_param():
    def __init__(self, model):
        self.model = model

    def statics_all(self, args):
        self.params = self.__getParam()
        num_layers = len(self.params)
        for l in range(num_layers):
            param = "l"+str(l+1)
            length, acc_weight, min_weight, max_weight = self.__statics(l, self.model.__dict__[param], args)
            average_weight = acc_weight/length
            distance = max_weight - min_weight
            print("Layer %d: Avg Weight=%f, Max Weight=%f, Min Weight=%f, Distance=%f" % (l, average_weight, max_weight, min_weight, distance))

    def __getParam(self):
        params = []       
        for key in self.model.__dict__.keys():
            if isinstance(self.model.__dict__[key], L.connection.linear.Linear):
                params.append(key)
        return params

    def __statics(self, l, params, args):
        weights = params.W

        acc_weight = 0.0
        min_w = 0.0
        max_w = 0.0

        for i in range(length):
            weight = weights[l][i].data
            acc_weight += float(weight)
            if float(weight) < min_w:
                min_w = float(weight)
            if float(weight) > max_w:
                max_w = float(weight)

        return (length, acc_weight, min_w, max_w)


# Main Function
def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--model_path', '-m', default='./my.model',
                        help='Model File Name and Path')
    args = parser.parse_args()

    # Set up a neural network
    model = MLP(args.unit, 10)

    # Load model    
    chainer.serializers.load_npz(args.model_path, model)

    # Make statics instance
    vp = statics_param(model)

    # Run statics
    vp.statics_all(args)


if __name__ == '__main__':
    main()