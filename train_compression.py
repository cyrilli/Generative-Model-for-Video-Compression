import sys
sys.path.append("./dcgan/")
from utils import *
sys.path.append("./vgg/")
import vgg_network

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os, time
from glob import glob
from random import shuffle

from model_compression import vanilla_encoder, dcgan_decoder
from utils_compression_model import LossCalculator

def main(_):
    ########################### Set Parameters ###########################
    dataset = 'celebA'
    train_size = np.inf
    num_epochs = 25
    sample_step = 500 # how often we plot and see the reconstruction results
    sample_size = 64 # the size of sample images
    save_step = 500 # how often we save the parameters of the network

    batch_size = 64
    original_size = 108 # original image size
    is_crop = True
    input_size = 64 # image size after crop
    c_dim = 3
    l = 0.5 # used to determine weight of pixel and perceptual loss
    lr = 0.0005 # learning rate

    dcgan_param_path = './dcgan/checkpoint/celebA_64_64/'
    checkpoint_dir = './checkpoint/'
    sample_dir = 'samples'
    vgg_path = './vgg/imagenet-vgg-verydeep-19.mat'
    ############################ Define Model ############################
    print('Building Model...')
    # feed input_img into encoder and get latent_var, and feed latent_var into decoder get output_img
    input_img = tf.placeholder(tf.float32, [batch_size, input_size, input_size, c_dim], name='input_img')
    print('Input shape: ', input_img.get_shape())
    encoder_net, _ = vanilla_encoder(input_img, z_dim=100)
    print('Latent shape: ', encoder_net.outputs.get_shape())
    decoder_net, _ = dcgan_decoder(encoder_net.outputs, image_size = input_size, c_dim=c_dim, batch_size=batch_size)
    print('Output shape: ', decoder_net.outputs.get_shape())
    print('Model successfully built!')
    #################### Define Loss and Training Ops ####################
    # pixel loss: mse
    #loss_pixel = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(decoder_net.outputs, input_img), [1, 2, 3]))
    loss_pixel = tf.nn.l2_loss(input_img - decoder_net.outputs) / (batch_size*input_size*input_size*c_dim)

    # computed by the forth convolutional layer of a Image-Net pretrained AlexNet
    # concat_img = tf.concat([input_img, decoder_net.outputs], axis = 0) # concat images along the first dimension
    # alexnet = alexnet_model(concat_img)
    # concat_features = alexnet.conv4
    # print('Shape of concat features: ', concat_features.get_shape())
    # # split the features into two partitions evenly
    # alexnet_features_input_img, alexnet_features_output_img = tf.split(concat_features, num_or_size_splits=2, axis=0)
    # loss_perceptual = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(alexnet_features_output_img, alexnet_features_input_img), [1, 2, 3]))
    vgg = vgg_network.VGG(vgg_path)
    loss_calculator = LossCalculator(vgg, decoder_net.outputs)
    loss_perceptual = loss_calculator.content_loss(input_img,content_layer='relu4_3',content_weight=1) / batch_size
    #loss_perceptual = tf.constant(0.0)
    # weighted sum of the two losses
    loss = l*loss_pixel + (1-l)*loss_perceptual

    train_param = encoder_net.all_params+decoder_net.all_params # update only the parameters of encoder network
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, var_list=train_param)

    ######################### Initialization Step ########################
    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess)

    # load trained parameters of dcgan_decoder
    print('Loading trained parameters of decoder network...')
    decoder_params = tl.files.load_npz(name=dcgan_param_path + 'net_g.npz')
    tl.files.assign_params(sess, decoder_params, decoder_net)
    print("Having loaded trained parameters of decoder network!")
    decoder_net.print_params()

    # load trained parameters of alexnet for extracting features
    #alexnet.load_initial_weights(sess)

    # Set the path to save parameters
    model_dir = "%s_%s_%s" % (dataset, batch_size, input_size)
    save_dir = os.path.join(checkpoint_dir, model_dir)
    tl.files.exists_or_mkdir(save_dir)
    tl.files.exists_or_mkdir(sample_dir)

    enc_compression_path = os.path.join(save_dir, 'enc_compression.npz')
    dec_compression_path = os.path.join(save_dir, 'dec_compression.npz')

    # get list of all training images' paths
    data_files = glob(os.path.join("./data", dataset, "*.jpg")) # returns a list of paths for images

    ############################## Train Model ###############################
    iter_counter = 0
    for epoch in range(num_epochs):
        ## shuffle data list
        shuffle(data_files)

        ## update sample files based on shuffled data
        sample_files = data_files[0:sample_size]
        sample = [get_image(sample_file, original_size, is_crop=is_crop, resize_w=input_size, is_grayscale = 0) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)
        print("[*] Sample images updated!")

        ## compute the number of batch per epoch
        batch_idxs = min(len(data_files), train_size) // batch_size

        for idx in xrange(0, batch_idxs):
            batch_files = data_files[idx*batch_size:(idx+1)*batch_size] # list, containing path of a batch

            ## get real images
            batch = [get_image(batch_file, original_size, is_crop=is_crop, resize_w=input_size, is_grayscale = 0)
                     for batch_file in batch_files]
            batch_images = np.array(batch).astype(np.float32)

            start_time = time.time()
            # updates the discriminator
            pix_loss, percep_loss, tot_loss, _ = sess.run([loss_pixel, loss_perceptual, loss, train_op], feed_dict={input_img: batch_images})

            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, pix_loss: %.8f, percep_loss: %.8f, tot_loss: %.8f" \
                    % (epoch, num_epochs, idx, batch_idxs, time.time() - start_time, pix_loss, percep_loss, tot_loss))

            iter_counter += 1
            # Save sample images and their reconstructions
            if np.mod(iter_counter, sample_step) == 0:
                # generate and visualize generated images
                recon_img = sess.run(decoder_net.outputs, feed_dict={input_img: sample_images})
                print('Shape of input sample is: ', sample_images.shape)
                print('Shape of recon sample is: ', recon_img.shape)
                save_sample = np.concatenate((sample_images, recon_img), axis=0)
                print('Shape of save_sample is ', save_sample.shape)
                tl.visualize.save_images(save_sample, [16, 8], './{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx))
                #print("[Sample] d_loss: %.8f, g_loss: %.8f" % (errD, errG))

            # Save network parameters
            if np.mod(iter_counter, save_step) == 0:
                # save current network parameters
                print("[*] Saving checkpoints...")
                tl.files.save_npz(encoder_net.all_params, name=enc_compression_path, sess=sess)
                tl.files.save_npz(decoder_net.all_params, name=dec_compression_path, sess=sess)
                print("[*] Saving checkpoints SUCCESS!")

if __name__ == '__main__':
    tf.app.run()