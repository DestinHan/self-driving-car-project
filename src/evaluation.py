import matplotlib.pyplot as plt
import numpy as np

def show_result(history, epochs):
	
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(epochs), history.history["loss"], label="Training Loss")
    plt.plot(np.arange(epochs), history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plt.savefig("outputs/loss_plot.png")
    plt.show()