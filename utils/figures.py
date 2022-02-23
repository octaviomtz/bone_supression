import matplotlib.pyplot as plt
import numpy as np

def xray_overview(source, target):
    plt.figure(figsize=(15,10))
    for i in range(3):
        ax = plt.subplot(2, 3, i+1)
        plt.imshow(source[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('Source')

        ax = plt.subplot(2, 3, i+4)
        plt.imshow(target[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('Target')
    plt.show()

def fig_loss(autoencoder_train, n_epoch):
    n = np.arange(0, n_epoch)
    plt.figure()
    plt.semilogy(n, autoencoder_train.history['loss'], label = 'Training Loss', color="#3F5D7D")
    plt.semilogy(n, autoencoder_train.history['val_loss'], label = 'Validation Loss', color='#883333')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('figures/loss.png')

def fig_prediction_xray(source_test, pred, target_test):
    plt.figure(figsize=(15,10))
    for i in range(5):
        ax = plt.subplot(3, 5, i+1)
        plt.imshow(source_test[i].reshape(128,128))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('Source_Test')

        ax = plt.subplot(3, 5, i+6)
        plt.imshow(pred[i].reshape(128,128))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('Predicted')
        
        ax = plt.subplot(3, 5, i+11)
        plt.imshow(target_test[i].reshape(128,128))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('Target')
    plt.show()