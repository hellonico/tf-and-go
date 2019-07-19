import sys
import tensorflow as tf
from dytb.inputs.predefined.MNIST import MNIST
from dytb.models.predefined.LeNetDropout import LeNetDropout
from dytb.train import train

def main():
    """main executes the operations described in the module docstring"""
    lenet = LeNetDropout()
    mnist = MNIST()

    info = train(
        model=lenet,
        dataset=mnist,
        hyperparameters={"epochs": 2},)

    checkpoint_path = info["paths"]["best"]

    with tf.Session() as sess:
        # Define a new model, import the weights from best model trained
        # Change the input structure to use a placeholder
        images = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name="input_")
        # define in the default graph the model that uses placeholder as input
        _ = lenet.get(images, mnist.num_classes)

        # The best checkpoint path contains just one checkpoint, thus the last is the best
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))

        # Create a builder to export the model
        builder = tf.saved_model.builder.SavedModelBuilder("export")
        # Tag the model in order to be capable of restoring it specifying the tag set
        # clear_device=True in order to export a device agnostic graph.
        builder.add_meta_graph_and_variables(sess, ["tag"], clear_devices=True)
        builder.save()

    return 0


if __name__ == '__main__':
    sys.exit(main())