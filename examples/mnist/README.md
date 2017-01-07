# Multi-Layer Perceptron for MNIST Classification

This is a minimal example to write a feed-forward net specialized for inference.

Optional arguments for running are as follows;

1. --gpu=N, N=-1: CPU, N>-1: GPU(s), default N=-1

2. --out='result': Specify output directory, default='result'

3. --unit=U, Number of Units, default U=1000

4. --num_inf=I, Number of Inferences, default=15

5. --fig_num_clm=C, Number of Columns for Plottig Figures, default=5


Requirement to run this script:
You need copy following statements to your example/mnist/train_mnist.py after trainer.run() statement:
    # Save model
    chainer.serializers.save_npz('my.model', trainer)


Acknowledgements:
Thanks for corochann-san, I used his/her code part of plotting figures and progress satus at running.
His/her code is here:
https://gist.github.com/corochann/180c29d0037d50ebceb295d8cafe1e3c
