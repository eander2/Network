import numpy as np
import random
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

import pandas as pd

random.seed()

class Activation(object):
    def __init__(self):
        return
    @staticmethod
    def transform(x):
        """Default to Selu"""
        _lambda = 1.0507
        alpha = 1.6732
        return 1.0 / (1.0 + np.exp(-x))
        # return _lambda * np.where(x >= 0.0, x, (alpha*np.exp(x) - alpha))
        # return np.where(x > 0.0, x, 0)

    #@staticmethod
    def activation_derivative(self, x):
        """Selu activation gradient"""
        _lambda = 1.0507
        alpha = 1.6732
        return self.transform(x) * (1 - self.transform(x))
        # return _lambda*np.where(x >= 0.0, 1.0, self.transform(x) + alpha)
        # return np.where(x > 0.0, 1.0, 0.0)

class Selu(Activation):
    @staticmethod
    def transform(x):
        """Default to Selu"""
        _lambda = 1.0507
        alpha = 1.6732
        return _lambda * np.where(x >= 0.0, x, (alpha*np.exp(x) - alpha))

    #@staticmethod
    def activation_derivative(self, x):
        """Selu activation gradient"""
        _lambda = 1.0507
        alpha = 1.6732
        return _lambda*np.where(x >= 0.0, 1.0, self.transform(x) + alpha)


class Softmax(Activation):
    @staticmethod
    def transform(x):
        """Default to Selu"""
        x_dmax = x - np.max(x)
        eps = np.exp(x_dmax)
        return eps/np.sum(eps)

    #@staticmethod
    def activation_derivative(self, x):
        """Selu activation gradient"""
        _lambda = 1.0507
        alpha = 1.6732
        return self.transform(x) * (1 - self.transform(x))
        # return _lambda*np.where(x >= 0.0, 1.0, self.transform(x) + alpha)
        # return np.where(x > 0.0, 1.0, 0.0)

class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).
        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.
        """
        return (a-y)


class generic_layer(object):
    def __init__(self, neuron_count, activation_class, **kwargs):
        self._parent = None
        self._child = None
        self._is_input = False
        self._neuron_count = neuron_count


        """Numerical parameters for layer.  Also includes those for backprop."""
        self.weights = None
        self.bias = None
        self.activity = None
        self.error = None
        self.activation = activation_class
        self.z = None
        self.weight_update = None
        self.bias_updates = None
        self.delta = None
        self.dropout = None
        self.dropout_mask = None

        #Array mask for missing data
        self.missing_mask = None
        self.missing_array = None
        self.fraction_missing = 0.0

        if 'activation' in kwargs:
            self._activation = kwargs['activation']
        if 'dropout' in kwargs:
            self.dropout = kwargs['dropout']

    def get_count(self):
        return self._neuron_count

    def Parent(self):
        return self._parent

    def Child(self):
        return self._child

    def set_parent(self, layer):
        self._parent = layer
        #self.weights = np.random.normal(0.0, 1.0/np.sqrt(layer.get_count()), (self._neuron_count, layer.get_count()))
        self.weights = 0.1*np.random.randn(self._neuron_count, layer.get_count())
        # self.bias = np.random.normal(0.0, 1.0/np.sqrt(layer.get_count()), (self._neuron_count, 1))
        # self.bias = np.random.randn(self._neuron_count, 1)
        self.bias = np.zeros(shape=(self._neuron_count, 1))


    def set_child(self, layer):
        self._child = layer

    def forward_propagation(self, x = None, is_backprop=False):
        w = self.weights
        missing_ratio = 0.0
        weight_ratio = 1.0
        if type(self.Parent()) == InputLayer:
            missing_array = np.ones(self.weights.shape)
            missing = np.isnan(self.Parent().activity)

            counter = 0.0
            for idx, val in enumerate(missing):
                if val:
                    missing_array[:, idx] = 0.0
                    counter += 1.0

            missing_ratio = counter/float(len(missing))
            weight_ratio = np.linalg.norm(w) / np.linalg.norm(self.weights)

            w = self.weights*missing_array


        #check activity for nans and replace with zero
        activities = self.Parent().activity
        activities[np.isnan(activities)] = 0

        dropout_mask = np.ones(shape = (self.get_count(),1))
        if self.dropout is not None and is_backprop:
            dropout_mask = np.random.binomial(1, 1.0-self.dropout, size=(self.get_count(),1)) / (1.0-self.dropout)

        self.dropout_mask = dropout_mask
        self.z = np.dot(w, activities)+self.bias
        # scale with missing ratio
        #self.z = self.z * dropout_mask/(1.0 - missing_ratio)
        # sclae with weight norm
        self.z = self.z * dropout_mask/(weight_ratio)
        xp = self.activation.transform(self.z)
        self.activity = xp

    def backwards_propagation(self):
        z_grad = self.activation.activation_derivative(self.z)
        delta = np.matmul(self._child.weights.transpose(), self._child.delta) * z_grad

        if self.dropout is not None:
            #delta = np.where(self.dropout_mask, 0.0, delta)
            delta *= self.dropout_mask

        self.delta = delta

        bias_update = delta

        weight_update = np.dot(delta, self.Parent().activity.transpose())

        return [weight_update, bias_update]


class InputLayer(generic_layer):
    def forward_propagation(self, x = None, is_backprop=False):
        self.missing_mask = np.isnan(x)
        # x[self.missing_mask] = 0
        self.activity = x

    def backwards_propagation(self):
        return


class FullyConnectedLayer(generic_layer):
    def void_function(self):
        return

class Optimizer(object):
    def __init__(self, learning_rate, lmbda, minibatch_size, **kwargs):
        self.model = None
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size

    def arrays_to_minibatch(self, X, Y):
        zlist = list(zip(X,Y))
        np.random.shuffle(zlist)
        X, Y = zip(*zlist)
        X_batches = [X[i:i+self.minibatch_size] for i in range(0, len(X), self.minibatch_size)]
        Y_batches = [Y[i:i + self.minibatch_size] for i in range(0, len(Y), self.minibatch_size)]
        return X_batches, Y_batches

    def optimize(self, X, Y, X_test = None, Y_test = None):
        X_batches, Y_batches = self.arrays_to_minibatch(X,Y)
        #ret = self.arrays_to_minibatch(X,Y)
        total_count = float(len(X))

        w_updates = []
        b_updates = []

        for idx in range(len(X_batches)):  # zip(X_batches, Y_batches):
            X_batch = X_batches[idx]
            Y_batch = Y_batches[idx]

            weight_update = [np.zeros(self.model.layers[idx].weights.shape) for idx in range(1, len(self.model.layers))]
            bias_update = [np.zeros(self.model.layers[idx].bias.shape) for idx in range(1, len(self.model.layers))]

            minibatch_correct = 0
            for x, y in zip(X_batch, Y_batch):
                dw, db = self.model.backprop(x, y)

                for idx, update in enumerate(dw):
                    weight_update[idx] += update

                for idx, update in enumerate(db):
                    bias_update[idx] += update

                xp = self.model.predict(x)
                #print("prediction argmax is {} and actual is {}".format(np.argmax(xp), np.argmax(y)))
                if np.argmax(xp) == np.argmax(y):
                    minibatch_correct += 1

            for idx in range(1, len(self.model.layers)):
                self.model.layers[idx].weights = (1.0) * \
                                                 self.model.layers[idx].weights - self.learning_rate*weight_update[idx-1]/float(len(Y_batch))
                self.model.layers[idx].biases = (1.0) * \
                                                 self.model.layers[idx].bias - self.learning_rate * bias_update[idx-1]/float(len(Y_batch))

            percent_correct = minibatch_correct/float(len(X_batch))
            print(f"The minibatch accuracy is {percent_correct}")


        training_correct = 0
        for x,y in zip(X,Y):
            y_predicted = self.model.predict(x)

            if np.argmax(y_predicted) == np.argmax(y):
                training_correct += 1

        training_accuracy = float(training_correct)/float(len(Y))
        print(f"Training accuracy is: {training_accuracy}")

        test_correct = 0
        if X_test is not None and Y_test is not None:
            for x,y in zip(X_test, Y_test):
                y_predicted = self.model.predict(x)

                if np.argmax(y_predicted) == np.argmax(y):
                    test_correct += 1

        validation_accuracy = float(test_correct)/float(len(Y_test))
        print(f"Test accuracy is: {validation_accuracy}")

        print("Finished Training Epoch.")
        return training_accuracy, validation_accuracy

class Sequential(object):
    def __init__(self):
        self.loss = None
        self.layers = []

    def addLayer(self, layer):
        self.layers.append(layer)

        if len(self.layers) > 1:
            self.layers[-2].set_child(self.layers[-1])
            self.layers[-1].set_parent(self.layers[-2])


    def predict(self, x, is_backprop=False):
        assert(len(self.layers) > 2)

        #self.layers[0].forward_propagation(x)
        self.layers[0].activity = x

        for idx in range(1, len(self.layers)):
            self.layers[idx].forward_propagation(is_backprop=is_backprop)

        return self.layers[-1].activity

    def backprop(self, x,y):
        """Forward Pass"""
        self.predict(x, is_backprop=True)

        """Backward Pass"""
        delta = self.loss.delta(self.layers[-1].z, self.layers[-1].activity, y)

        self.layers[-1].delta = delta

        b1 = delta

        w1 = np.dot(delta, self.layers[-1].Parent().activity.transpose())

        weight_update = [w1]
        bias_update = [b1]

        for idx in range(2, len(self.layers)):
            w, b = self.layers[-idx].backwards_propagation()
            weight_update.append(w)
            bias_update.append(b)

        weight_update = reversed(weight_update)
        bias_update = reversed(bias_update)

        return weight_update, bias_update


    #def fit(self, X, Y,batch_size, num_epochs, optimizer, loss, **kwargs):


def corrupt_data(data, fraction=0.1):
    mask = np.random.choice([True, False], size=np.prod(data.shape), p=[fraction, 1.0-fraction])
    mask = mask.reshape(data.shape)
    data[mask] = np.nan
    return data

def corrupt_scanline(data, lines = None, fraction = 0.0, image_corrupt_prob=0.0):
    if lines is None:
        lines = np.random.choice([True, False], size=data.shape[0], p=[fraction, 1.0 - fraction])

    # check to see if image is corrupted:
    # corrupt = np.random.choice([True, False], size=(1,), p=[image_corrupt_prob, 1.0 - image_corrupt_prob])

    if random.random() > image_corrupt_prob:
        return data, lines

    for idx, val in enumerate(lines):
        if val:
            data[idx, :] = np.nan

    return data, lines

if __name__ == "__main__":
    mnist = input_data.read_data_sets('./mnist/', one_hot=True)
    images, labels = mnist.test.images, mnist.test.labels

    zlist = list(zip(images, labels))
    np.random.shuffle(zlist)
    images, labels = zip(*zlist)

    print(images[0].shape)
    training_data = []
    training_label = []

    corrupt_lines = None
    for image, label in zip(images, labels):
        #training_data.append((image.ravel().reshape(784, 1), label.reshape(10, 1)))
        i = image.ravel().reshape(784,1)
        # i = corrupt_data(i, fraction=0.5)

        i, corrupt_lines = corrupt_scanline(image.reshape(28,28), lines = corrupt_lines, fraction = 0.3, image_corrupt_prob = 0.5)
        #plt.imshow(i)
        #plt.show()
        training_data.append(i.ravel().reshape(784, 1))

        training_label.append(label.ravel().reshape(10, 1))


    model = Sequential()
    model.loss = CrossEntropyCost()
    model.addLayer(InputLayer(784, activation_class=Activation()))
    model.addLayer(FullyConnectedLayer(128, activation_class=Activation(), dropout=0.5))
    model.addLayer(FullyConnectedLayer(64, activation_class=Activation(), dropout=0.5))
    model.addLayer(FullyConnectedLayer(10, activation_class = Softmax()))

    opt = Optimizer(0.01, 0.0, 128)
    opt.model = model

    training_accuracy = []
    test_accuracy = []
    epoch = []

    holdout = 5000
    for idx in range(1000):
        print(f"Starting epoch {idx}.")
        train, test = opt.optimize(training_data[:-holdout], training_label[:-holdout], training_data[-holdout:], training_label[-holdout:])
        training_accuracy.append(train)
        test_accuracy.append(test)
        epoch.append(idx)




    df = pd.DataFrame()

    df['Training_Epoch'] = epoch
    df['Test_Accuracy'] = test_accuracy
    df['Training_Accuracy'] = training_accuracy

    #save dataframe to disk for analysis

