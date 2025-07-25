import struct
import numpy as np
import math
import random

#Convolutional layer backward propagation
# Numpy version: compute with np.dot
def conv_backward(dZ, cache):

    (A_prev, W, b, hparameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape

    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    (m, n_H, n_W, n_C) = dZ.shape
    
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))
    
    if pad != 0:
        A_prev_pad = zero_pad(A_prev, pad)
        dA_prev_pad = zero_pad(dA_prev, pad)
    else:
        A_prev_pad = A_prev
        dA_prev_pad = dA_prev
    
    for h in range(n_H):                    # loop over vertical axis of the output volume
        for w in range(n_W):                # loop over horizontal axis of the output volume
            # Find the corners of the current "slice"
            vert_start, horiz_start  = h*stride, w*stride
            vert_end,   horiz_end    = vert_start+f, horiz_start+f
            
            # Use the corners to define the slice from a_prev_pad
            A_slice = A_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end, :]
            
            # Update gradients for the window and the filter's parameters
            dA_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end, :] += np.transpose(np.dot(W, dZ[:, h, w, :].T), (3,0,1,2))

            dW += np.dot(np.transpose(A_slice, (1,2,3,0)), dZ[:, h, w, :])
            db += np.sum(dZ[:, h, w, :], axis=0)
            
    # Set dA_prev to the unpaded dA_prev_pad
    dA_prev = dA_prev_pad if pad == 0 else dA_prev_pad[:,pad:-pad, pad:-pad, :]
        
    # Making sure your output shape is correct
    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    
    return dA_prev, dW, db

#Pooling layer backpropagation
def pool_backward(dA, cache, mode):
    
    (A_prev, hparameters) = cache
    
    stride = hparameters["stride"]
    f = hparameters["f"]

    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape #256,28,28,6
    m, n_H, n_W, n_C = dA.shape                    #256,14,14,6
    
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev)) #256,28,28,6
        
    for h in range(n_H):                    # loop on the vertical axis
        for w in range(n_W):                # loop on the horizontal axis
            # Find the corners of the current "slice"
            vert_start, horiz_start  = h*stride, w*stride
            vert_end,   horiz_end    = vert_start+f, horiz_start+f
            
            # Compute the backward propagation in both modes.
            if mode == "max":
                A_prev_slice = A_prev[:, vert_start: vert_end, horiz_start: horiz_end, :] 
                A_prev_slice = np.transpose(A_prev_slice, (1,2,3,0))
                mask = A_prev_slice==A_prev_slice.max((0,1))           
                mask = np.transpose(mask, (3,2,0,1))                   
                dA_prev[:, vert_start: vert_end, horiz_start: horiz_end, :] \
                      += np.transpose(np.multiply(dA[:, h, w, :][:,:,np.newaxis,np.newaxis],mask), (0,2,3,1))

            elif mode == "average":
                da = dA[:, h, w, :][:,np.newaxis,np.newaxis,:]  #256*1*1*6
                dA_prev[:, vert_start: vert_end, horiz_start: horiz_end, :] += np.repeat(np.repeat(da, 2, axis=1), 2, axis=2)/f/f
    
    assert(dA_prev.shape == A_prev.shape)
    return dA_prev

#Layers Classes

# padding for the matrix of images
def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, ), (pad, ), (pad, ), (0, )), 'constant', constant_values=(0, 0))    
    return X_pad

# normalization of the input images
def normalize(image, mode='lenet5'):
    image -= image.min()
    image = image / image.max()
    # range = [0,1]
    if mode == '0p1':
        return image
    # range = [-1,1]
    elif mode == 'n1p1':
        image = image * 2 - 1
    # range = [-0.1,1.175]   
    elif mode == 'lenet5':
        image = image * 1.275 - 0.1
    return image

# initialization of the weights & bias
def initialize(kernel_shape, mode='Fan-in'):
    b_shape = (1,1,1,kernel_shape[-1]) if len(kernel_shape)==4 else (kernel_shape[-1],)
    if mode == 'Gaussian_dist':
        mu, sigma = 0, 0.1
        weight = np.random.normal(mu, sigma,  kernel_shape) 
        bias   = np.ones(b_shape)*0.01
        
    elif mode == 'Fan-in': #original init. in the paper
        Fi = np.prod(kernel_shape)/kernel_shape[-1]
        weight = np.random.uniform(-2.4/Fi, 2.4/Fi, kernel_shape)    
        bias   = np.ones(b_shape)*0.01     
    return weight, bias

# update for the weights
def update(weight, bias, dW, db, vw, vb, lr, momentum=0, weight_decay=0):
    vw_u = momentum*vw - weight_decay*lr*weight - lr*dW
    vb_u = momentum*vb - weight_decay*lr*bias   - lr*db
    weight_u = weight + vw_u
    bias_u   = bias   + vb_u
    return weight_u, bias_u, vw_u, vb_u 

# return random-shuffled mini-batches
def random_mini_batches(image, label, mini_batch_size = 256, one_batch=False):
    m = image.shape[0]                  # number of training examples
    mini_batches = []
    
    # Shuffle (image, label)
    permutation = list(np.random.permutation(m))
    shuffled_image = image[permutation,:,:,:]
    shuffled_label = label[permutation]
    
    # extract only one batch
    if one_batch:
        mini_batch_image = shuffled_image[0: mini_batch_size,:,:,:]
        mini_batch_label = shuffled_label[0: mini_batch_size]
        return (mini_batch_image, mini_batch_label)

    # Partition (shuffled_image, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_image = shuffled_image[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_label = shuffled_label[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_image, mini_batch_label)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_image = shuffled_image[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_label = shuffled_label[num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_image, mini_batch_label)
        mini_batches.append(mini_batch)
    
    return mini_batches

#Convolutional layer utility
def conv_SDLM(dZ, cache):
    (A_prev, W, b, hparameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    (m, n_H, n_W, n_C) = dZ.shape
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))
    
    if pad != 0:
        A_prev_pad = zero_pad(A_prev, pad)
        dA_prev_pad = zero_pad(dA_prev, pad)
    else:
        A_prev_pad = A_prev
        dA_prev_pad = dA_prev
    
    for h in range(n_H):                    # loop over vertical axis of the output volume
        for w in range(n_W):                # loop over horizontal axis of the output volume
            # Find the corners of the current "slice"
            vert_start, horiz_start  = h*stride, w*stride
            vert_end,   horiz_end    = vert_start+f, horiz_start+f
            
            # Use the corners to define the slice from a_prev_pad
            A_slice = A_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end, :]
            
            # Update gradients for the window and the filter's parameters
            dA_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end, :] += np.transpose(np.dot(np.power(W,2), dZ[:, h, w, :].T), (3,0,1,2))

            dW += np.dot(np.transpose(np.power(A_slice,2), (1,2,3,0)), dZ[:, h, w, :])
    # Set dA_prev to the unpaded dA_prev_pad
    dA_prev = dA_prev_pad if pad == 0 else dA_prev_pad[:,pad:-pad, pad:-pad, :]
    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    return dA_prev, dW

#bitmaps
bitmap = np.zeros((10,84))
bitmap[0]=np.array([
    [-1, +1, +1, +1, +1, +1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] + \
    [-1, -1, +1, +1, +1, -1, -1] + \
    [-1, +1, +1, -1, +1, +1, -1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [-1, +1, +1, -1, +1, +1, -1] + \
    [-1, -1, +1, +1, +1, -1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] \
])
bitmap[1]=np.array([
    [-1, -1, -1, +1, +1, -1, -1] + \
    [-1, -1, +1, +1, +1, -1, -1] + \
    [-1, +1, +1, +1, +1, -1, -1] + \
    [-1, -1, -1, +1, +1, -1, -1] + \
    [-1, -1, -1, +1, +1, -1, -1] + \
    [-1, -1, -1, +1, +1, -1, -1] + \
    [-1, -1, -1, +1, +1, -1, -1] + \
    [-1, -1, -1, +1, +1, -1, -1] + \
    [-1, -1, -1, +1, +1, -1, -1] + \
    [-1, +1, +1, +1, +1, +1, +1] + \
    [-1, -1, -1, -1, -1, -1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] \
])
bitmap[2]=np.array([
    [-1, +1, +1, +1, +1, +1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] + \
    [-1, +1, +1, +1, +1, +1, -1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [+1, -1, -1, -1, -1, +1, +1] + \
    [-1, -1, -1, -1, +1, +1, -1] + \
    [-1, -1, +1, +1, +1, -1, -1] + \
    [-1, +1, +1, -1, -1, -1, -1] + \
    [+1, +1, -1, -1, -1, -1, -1] + \
    [+1, +1, +1, +1, +1, +1, +1] + \
    [-1, -1, -1, -1, -1, -1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] \
])
bitmap[3]=np.array([
    [+1, +1, +1, +1, +1, +1, +1] + \
    [-1, -1, -1, -1, -1, +1, +1] + \
    [-1, -1, -1, -1, +1, +1, -1] + \
    [-1, -1, -1, +1, +1, -1, -1] + \
    [-1, -1, +1, +1, +1, +1, -1] + \
    [-1, -1, -1, -1, -1, +1, +1] + \
    [-1, -1, -1, -1, -1, +1, +1] + \
    [-1, -1, -1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [-1, +1, +1, +1, +1, +1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] \
])
bitmap[4]=np.array([
    [-1, +1, +1, +1, +1, +1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] + \
    [-1, +1, +1, -1, -1, +1, +1] + \
    [-1, +1, +1, -1, -1, +1, +1] + \
    [+1, +1, +1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, +1, +1, +1] + \
    [-1, +1, +1, +1, +1, +1, +1] + \
    [-1, -1, -1, -1, -1, +1, +1] + \
    [-1, -1, -1, -1, -1, +1, +1] \
])
bitmap[5]=np.array([
    [-1, +1, +1, +1, +1, +1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] + \
    [+1, +1, +1, +1, +1, +1, +1] + \
    [+1, +1, -1, -1, -1, -1, -1] + \
    [+1, +1, -1, -1, -1, -1, -1] + \
    [-1, +1, +1, +1, +1, -1, -1] + \
    [-1, -1, +1, +1, +1, +1, -1] + \
    [-1, -1, -1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [-1, +1, +1, +1, +1, +1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] \
])
bitmap[6]=np.array([
    [-1, -1, +1, +1, +1, +1, -1] + \
    [-1, +1, +1, -1, -1, -1, -1] + \
    [+1, +1, -1, -1, -1, -1, -1] + \
    [+1, +1, -1, -1, -1, -1, -1] + \
    [+1, +1, +1, +1, +1, +1, -1] + \
    [+1, +1, +1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [+1, +1, +1, -1, -1, +1, +1] + \
    [-1, +1, +1, +1, +1, +1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] \
])
bitmap[7]=np.array([
    [+1, +1, +1, +1, +1, +1, +1] + \
    [-1, -1, -1, -1, -1, +1, +1] + \
    [-1, -1, -1, -1, -1, +1, +1] + \
    [-1, -1, -1, -1, +1, +1, -1] + \
    [-1, -1, -1, +1, +1, -1, -1] + \
    [-1, -1, -1, +1, +1, -1, -1] + \
    [-1, -1, +1, +1, -1, -1, -1] + \
    [-1, -1, +1, +1, -1, -1, -1] + \
    [-1, -1, +1, +1, -1, -1, -1] + \
    [-1, -1, +1, +1, -1, -1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] \
])
bitmap[8]=np.array([
    [-1, +1, +1, +1, +1, +1, -1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [-1, +1, +1, +1, +1, +1, -1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [-1, +1, +1, +1, +1, +1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] \
])
bitmap[9]=np.array([
    [-1, +1, +1, +1, +1, +1, -1] + \
    [+1, +1, -1, -1, +1, +1, +1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, +1, +1, +1] + \
    [-1, +1, +1, +1, +1, +1, +1] + \
    [-1, -1, -1, -1, -1, +1, +1] + \
    [-1, -1, -1, -1, -1, +1, +1] + \
    [-1, -1, -1, -1, +1, +1, -1] + \
    [-1, +1, +1, +1, +1, -1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] \
])

def rbf_init_weight():
    return bitmap

