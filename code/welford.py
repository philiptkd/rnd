# online algorithm for updating the mean and variance of a stream of data
# from https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm
class Welford():
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.mean = 0
        self.M2 = 0

    # for a new value newValue, compute the new count, new mean, the new M2.
    # mean accumulates the mean of the entire dataset
    # M2 aggregates the squared distance from the mean
    # count aggregates the number of samples seen so far
    def update(self, newValue):
        self.count += 1 
        delta = newValue - self.mean
        self.mean += delta / self.count
        delta2 = newValue - self.mean
        self.M2 += delta * delta2
        self.var = self.M2 / self.count
