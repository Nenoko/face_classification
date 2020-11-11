import matplotlib.pyplot as plt
import torch
import matplotlib
matplotlib.use('Agg')

training_loss = torch.load('./training_loss.pkl')
test_loss = torch.load('./test_loss.pkl', 'wb')
training_accuracy = torch.load('./training_accuracy.pkl')
test_accuracy = torch.load('./test_accuracy.pkl')


def draw(data, title, ylabel):
    xlabel = 'epoch'
    label = [(i+1) for i, _ in enumerate(data)]

    plt.plot(label, data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if "accuracy" in title:
        plt.ylim = (min(data) - 0.5, max(data) + 0.5)
    plt.savefig("{}.png".format(title))
    plt.gca().clear()


draw(training_loss, "training_loss", "loss")
draw(test_loss, "test_loss", "loss")
draw(training_accuracy, "training_accuracy", "accuracy[%]")
draw(test_accuracy, "test_accuracy", "accuracy[%]")
