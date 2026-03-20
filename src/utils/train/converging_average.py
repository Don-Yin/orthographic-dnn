class ExpMovingAverage:
    """tracks loss with exponential moving average for early stopping."""

    def __init__(self, start, alpha=0.5):
        self.avg = start
        self.alpha = alpha

    def __call__(self, value):
        self.avg = self.alpha * value + (1 - self.alpha) * self.avg
        return self


if __name__ == "__main__":
    moving_average = ExpMovingAverage(10)
    moving_average(5)
    moving_average(3)
    moving_average(2)
    moving_average(1)
    moving_average(0.3)
    print(moving_average.avg)
