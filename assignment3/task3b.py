import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer
import ExampleModel




def create_plots(trainer1: Trainer, trainer2: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(
        trainer1.train_history["loss"], 
        label="Training loss SGD", 
        npoints_to_average=10
    )
    utils.plot_loss(trainer1.validation_history["loss"], 
                    label="Validation loss SGD")
    utils.plot_loss(
        trainer2.train_history["loss"], 
        label="Training loss Adagrad", 
        npoints_to_average=10
    )
    utils.plot_loss(trainer2.validation_history["loss"], 
                    label="Validation loss Adagrad")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer1.validation_history["accuracy"], label="Validation Accuracy SGD")
    utils.plot_loss(trainer2.validation_history["accuracy"], label="Validation Accuracy Adagrad")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result!
    utils.set_seed(0)
    print(f"Using device: {utils.get_device()}")
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    
    model1 = ExampleModel(image_channels=3, num_classes=10)
    model2 = ExampleModel(image_channels=3, num_classes=10)
    
    trainer1 = Trainer(
        batch_size, learning_rate, early_stop_count, epochs1, model1, dataloaders
    )
    trainer2 = Trainer(
        batch_size, learning_rate, early_stop_count, epochs1, model2, dataloaders
    )
    
    trainer1.optimizer = torch.optim.SGD(trainer1.model.parameters(), trainer1.learning_rate)
    trainer2.optimizer = torch.optim.Adagrad(trainer2.model.parameters(), lr=trainer2.learning_rate, weight_decay=1e-5)
    
    trainer1.train()
    trainer2.train()
    

    create_plots(trainer1, trainer2, "task3b") 
    
    # test accuracies
    trainer1.evaluate_on_test()
    trainer2.evaluate_on_test()

if __name__ == "__main__":
    main()
