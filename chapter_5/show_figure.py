# show results
import matplotlib.pyplot as plt


def show_figure(his):
    acc = his.history['acc']
    val_acc = his.history['val_acc']
    loss = his.history['loss']
    val_loss = his.history['val_loss']

    epochs = range(1, len(acc)+1)
    plt.plot(epochs, acc, 'bo', label='training acc')
    plt.plot(epochs, val_acc, 'b', label='validation acc')
    plt.title('training and validation accuracy')
    plt.legend()

    plt.figure()
    plt.plot(epochs, loss, 'ro', label='traning loss')
    plt.plot(epochs, val_loss, 'r', label='validation loss')
    plt.title('training and validation loss')
    plt.legend()
    plt.show()


def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous*factor+point*(1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def draw_smooth_curve(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, smooth_curve(acc), 'bo', label='smoothed training acc')
    plt.plot(epochs, smooth_curve(val_acc), 'b', label='smoothed validation acc')
    plt.title('training and validation accuracy')
    plt.legend()

    plt.figure()
    plt.plot(epochs, smooth_curve(loss), 'ro', label='smoothed traning loss')
    plt.plot(epochs, smooth_curve(val_loss), 'r', label='smoothed validation loss')
    plt.title('training and validation loss')
    plt.legend()
    plt.show()