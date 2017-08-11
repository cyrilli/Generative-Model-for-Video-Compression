import tensorflow as tf
import numpy as np
from functools import reduce

class LossCalculator:

    def __init__(self, vgg, stylized_image):
        self.vgg = vgg
        self.transform_loss_net = vgg.net(vgg.preprocess(stylized_image))

    def content_loss(self, content_input_batch, content_layer, content_weight):
        content_loss_net = self.vgg.net(self.vgg.preprocess(content_input_batch))
        return tf.nn.l2_loss(content_loss_net[content_layer] - self.transform_loss_net[content_layer]) / (_tensor_size(content_loss_net[content_layer]))

    def style_loss(self, style_image, style_layers, style_weight):
        style_image_placeholder = tf.placeholder('float', shape=style_image.shape)
        style_loss_net = self.vgg.net(style_image_placeholder)

        with tf.Session() as sess:
            style_loss = 0
            style_preprocessed = self.vgg.preprocess(style_image)

            for layer in style_layers:
                style_image_gram = self._calculate_style_gram_matrix_for(style_loss_net,
                                                                   style_image_placeholder,
                                                                   layer,
                                                                   style_preprocessed)

                input_image_gram = self._calculate_input_gram_matrix_for(self.transform_loss_net, layer)

                style_loss += (2 * tf.nn.l2_loss(input_image_gram - style_image_gram) / style_image_gram.size)

            return style_weight * (style_loss)

    def tv_loss(self, image, tv_weight):
        # total variation denoising
        shape = tuple(image.get_shape().as_list())
        tv_y_size = _tensor_size(image[:,1:,:,:])
        tv_x_size = _tensor_size(image[:,:,1:,:])
        tv_loss = tv_weight * 2 * (
                (tf.nn.l2_loss(image[:,1:,:,:] - image[:,:shape[1]-1,:,:]) /
                    tv_y_size) +
                (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:shape[2]-1,:]) /
                    tv_x_size))

        return tv_loss

    def _calculate_style_gram_matrix_for(self, network, image, layer, style_image):
        image_feature = network[layer].eval(feed_dict={image: style_image})
        image_feature = np.reshape(image_feature, (-1, image_feature.shape[3]))
        return np.matmul(image_feature.T, image_feature) / image_feature.size

    def _calculate_input_gram_matrix_for(self, network, layer):
        image_feature = network[layer]
        batch_size, height, width, number = map(lambda i: i.value, image_feature.get_shape())
        size = height * width * number
        image_feature = tf.reshape(image_feature, (batch_size, height * width, number))
        return tf.matmul(tf.transpose(image_feature, perm=[0,2,1]), image_feature) / size


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)