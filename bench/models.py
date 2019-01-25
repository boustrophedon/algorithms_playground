import numpy as np

from sklearn import linear_model

# An attempt to do "automated bench testing" i.e. checking that a given function has a given performance characteristic.

# First, generate benchmarks using timeit.timeit. Then, fit the (input, runtime) points to a given model and assert that its R^2 is high enough. See algo/bench/merge.py for an example.

# TODO: use actual statistical tests instead of just looking at r-squared. currently we validate the parameters as a check for poor fit because you can get a high r-squared with an incorrect model, but I'm sure there are models where the parameters look good and the r-squared is also good.


class BigOModel:
    def fit(self, x, y):
        x = self.transform(np.array(x))
        y = np.array(y)

        self.model = linear_model.LinearRegression()
        self.model.fit(x, y)

        self.params = (self.model.coef_, self.model.intercept_)
        self.score = self.model.score(x, y)

    def predict(self, x):
        return self.func(x, *self.params)

    # Very basic sanity check for goodness of fit
    # If an algorithm runs in O(f(x)) time, it generally doesn't run in less than 1*f(x) actual steps.
    # Of course, it can: `def f(n): for _ in range(0,0.4n): pass;` runs in 0.4n
    # time. I bet you could come up with a recursive algorithm that has a
    # recurrance relationship that you could put into the master theorem and
    # get a leading coefficient <1. It's just very unlikely to actually happen
    # when it's actually implemented in code.
    def assert_good_params(self):
        assert (self.params[0] > 1).all(), self.params
        assert self.params[1] > 0, self.params

    def assert_bad_params(self):
        assert (self.params[0] < 1).any() or self.params[1] < 0


class Onlogn(BigOModel):
    def transform(self, x):
        return np.multiply(x, np.log2(x)).reshape(-1, 1)

    def func(self, x, a, b):
        return a * x * np.log2(x) + b


class Onsquared(BigOModel):
    def transform(self, x):
        return np.multiply(x, x).reshape(-1, 1)

    def func(self, x, a, b):
        return a * x * x + b


def test_model_nlogn():
    np.random.seed(1766)

    x = np.linspace(1, 100)

    model = Onlogn()
    y_true = model.func(x, 2, 5)

    y_noise = y_true + 0.2 * np.random.normal(size=x.size)

    model.fit(x, y_noise)

    print(model.params)
    print(model.score)

    model.assert_good_params()
    assert model.score > 0.99


def test_model_nsquared():
    np.random.seed(1766)

    x = np.linspace(1, 10000)

    model = Onsquared()
    y_true = model.func(x, 10, 100)

    y_noise = y_true + 0.2 * np.random.normal(size=x.size)

    model.fit(x, y_noise)

    print(model.params)
    print(model.score)

    model.assert_good_params()
    assert model.score > 0.99


# Test that params are bad with
def test_model_incorrect_nsquared():
    np.random.seed(1766)

    x = np.linspace(1, 10000)

    model = Onsquared()
    model_nlogn = Onlogn()
    y = model_nlogn.func(x, 1, 0)

    model.fit(x, y)

    print(model.params)
    print(model.score)

    model.assert_bad_params()


def test_model_incorrect_nlogn():
    np.random.seed(1766)

    x = np.linspace(1, 10000)
    y = np.linspace(1, 10000)

    model = Onlogn()

    model.fit(x, y)

    print(model.params)
    print(model.score)

    model.assert_bad_params()
