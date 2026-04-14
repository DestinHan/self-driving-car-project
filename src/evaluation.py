import matplotlib.pyplot as plt
import numpy as np

def show_result(history, epochs=None):
    ran_epochs = len(history.history["loss"])
    x = np.arange(ran_epochs)

    plt.figure(figsize=(8, 5))
    plt.plot(x, history.history["loss"], label="Training Loss")
    plt.plot(x, history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plt.savefig("outputs/loss_plot.png")
    plt.show()