import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = 0.9  # Task 3 hyperparameter
    
    shuffle_data = True
    use_improved_sigmoid = True
    use_improved_weight_init = False
    use_momentum = False
    use_relu = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    # Example created for comparing with and without shuffling.
    # For comparison, show all loss/accuracy curves in the same plot
    # YOU CAN DELETE EVERYTHING BELOW!

    shuffle_data = True
    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False
    use_relu = False

    # Train a new model with new parameters
    model_no_shuffle = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_shuffle = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_no_shuffle, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_no_shuffle, val_history_no_shuffle = trainer_shuffle.train(
        num_epochs)

    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history["loss"],
                    "Task 3 Model", npoints_to_average=10)
    utils.plot_loss(
        train_history_no_shuffle["loss"], "Task 3 Model - No use_improved_sigmoid", npoints_to_average=10)
    plt.ylim([0, 0.4])
    plt.subplot(1, 2, 2)
    plt.ylim([0.85, 0.95])
    utils.plot_loss(val_history["accuracy"], "Task 3 Model")
    utils.plot_loss(
        val_history_no_shuffle["accuracy"], "Task 3 Model - No use_improved_sigmoid")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig(f"task3c_train_loss_use_improved_sigmoid.png")
    plt.show()
       #############
        
    shuffle_data = True
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = False
    use_relu = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    # Example created for comparing with and without shuffling.
    # For comparison, show all loss/accuracy curves in the same plot
    # YOU CAN DELETE EVERYTHING BELOW!

    shuffle_data = True
    use_improved_sigmoid = True
    use_improved_weight_init = False
    use_momentum = False
    use_relu = False

    # Train a new model with new parameters
    model_no_shuffle = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_shuffle = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_no_shuffle, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_no_shuffle, val_history_no_shuffle = trainer_shuffle.train(
        num_epochs)

    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history["loss"],
                    "Task 3 Model", npoints_to_average=10)
    utils.plot_loss(
        train_history_no_shuffle["loss"], "Task 3 Model - No use_improved_weight_init", npoints_to_average=10)
    plt.ylim([0, 0.4])
    plt.subplot(1, 2, 2)
    plt.ylim([0.85, 0.95])
    utils.plot_loss(val_history["accuracy"], "Task 3 Model")
    utils.plot_loss(
        val_history_no_shuffle["accuracy"], "Task 3 Model - No use_improved_weight_init")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig(f"task3c_train_loss_use_improved_weight_init.png")
    plt.show()
    
    ###########@
    shuffle_data = True
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True
    use_relu = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    # Example created for comparing with and without shuffling.
    # For comparison, show all loss/accuracy curves in the same plot
    # YOU CAN DELETE EVERYTHING BELOW!

    shuffle_data = True
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = False
    use_relu = False

    # Train a new model with new parameters
    model_no_shuffle = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_shuffle = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_no_shuffle, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_no_shuffle, val_history_no_shuffle = trainer_shuffle.train(
        num_epochs)

    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history["loss"],
                    "Task 3 Model", npoints_to_average=10)
    utils.plot_loss(
        train_history_no_shuffle["loss"], "Task 3 Model - No use_momentum", npoints_to_average=10)
    plt.ylim([0, 0.4])
    plt.subplot(1, 2, 2)
    plt.ylim([0.85, 0.95])
    utils.plot_loss(val_history["accuracy"], "Task 3 Model")
    utils.plot_loss(
        val_history_no_shuffle["accuracy"], "Task 3 Model - No use_momentum")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig(f"task3c_train_loss_use_momentum.png")
    plt.show()

if __name__ == "__main__":
    main()
