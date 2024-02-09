import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    # TODO: Implement this function (task 3c)
    y_hat = model.forward(X)
    y_hat_digit = np.argmax(y_hat, axis=1)
    y_hat_binary = one_hot_encode(y_hat_digit, targets.shape[1])
    accuracy = np.mean(np.sum(y_hat_binary == targets, axis=1) == 10)
    return accuracy


class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        # TODO: Implement this function (task 3b)
        #self.model.w = np.zeros((self.model.I, self.model.num_outputs))
        self.model.zero_grad()
        y_hat = self.model.forward(X_batch)
        self.model.backward(X_batch, y_hat, Y_batch)
        self.model.w = self.model.w - self.learning_rate * self.model.grad
        loss = cross_entropy_loss(Y_batch, y_hat)
        return loss

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
            accuracy_train (float): Accuracy on train dataset
            accuracy_val (float): Accuracy on the validation dataset
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(self.Y_val, logits)

        accuracy_train = calculate_accuracy(
            self.X_train, self.Y_train, self.model)
        accuracy_val = calculate_accuracy(
            self.X_val, self.Y_val, self.model)
        return loss, accuracy_train, accuracy_val


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.


    

    # Intialize model
    model = SoftmaxModel(l2_reg_lambda)
    # Train model
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    plt.ylim([0.2, .8])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task3b_softmax_train_loss.png")
    plt.show()

    # Plot accuracy
    plt.ylim([0.89, .93])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3b_softmax_train_accuracy.png")
    plt.show()
    
    
    # Train a model with L2 regularization (task 4b)


    model1 = SoftmaxModel(l2_reg_lambda=1.0)
    trainer = SoftmaxTrainer(
        model1, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg01, val_history_reg01 = trainer.train(num_epochs)
    # You can finish the rest of task 4 below this point.

    # Plotting of softmax weights (Task 4b)
    # Plot weight without regularization
    no_biais0 = model.w[:-1]
    images0 = [no_biais0[:,i].reshape(28,28) for i in range(10)]
    weight0 = [[images0[j//28][i][j%28] for i in range(28)] for j in range(28*10)]
    weight0 = np.array(weight0).T
    # Plot weight with regularization
    no_biais1 = model1.w[:-1]
    images1 = [no_biais1[:,i].reshape(28,28) for i in range(10)]
    weight1 = [[images1[j//28][i][j%28] for i in range(28)] for j in range(28*10)]
    weight1 = np.array(weight1).T
    
    # Digits on the same plot
    weight = np.concatenate((weight0, weight1), axis=0)
    plt.imsave("task4b_softmax_weight.png", weight, cmap="gray")

   
    
    
    
    
    
    # Plotting of accuracy for difference values of lambdas (task 4c)
    l2_lambdas = [1, .1, .01, .001]
    val_history_l2 = []
    weights = []
    for l2_reg_lambda in l2_lambdas:
        model = SoftmaxModel(l2_reg_lambda)
        # Train model
        trainer = SoftmaxTrainer(
            model, learning_rate, batch_size, shuffle_dataset,
            X_train, Y_train, X_val, Y_val,
        )
        train_history, val_history = trainer.train(num_epochs)
        val_history_l2.append(val_history["accuracy"])
        weights.append(model.w)
        
        



    # Plot accuracy
    plt.ylim([0, 1])
    for i, l2_reg_lambda in enumerate(l2_lambdas):
        utils.plot_loss(val_history_l2[i], f"Validation Accuracy, lambda={l2_reg_lambda}")
        
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    
    plt.savefig("task4c_l2_reg_accuracy.png")
    plt.show()
    # Task 4d - Plotting of the l2 norm for each weight

    x = l2_lambdas
    y = [np.linalg.norm(weight) for weight in weights]
    
    plt.plot(x, y)
    plt.xlabel("L2 norm")
    plt.ylabel("Lambda")
    plt.savefig("task4d_l2_reg_norms.png")
    plt.show()
    
if __name__ == "__main__":
    main()
