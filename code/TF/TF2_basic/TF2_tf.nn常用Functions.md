#
```

```
# Functions
```


```

```
elu(...): Computes exponential linear: exp(features) - 1 if < 0, features otherwise.
crelu(...): Computes Concatenated ReLU.
leaky_relu(...): Compute the Leaky ReLU activation function.
log_softmax(...): Computes log softmax activations.
relu(...): Computes rectified linear: max(features, 0).
relu6(...): Computes Rectified Linear 6: min(max(features, 0), 6).
```

```
atrous_conv2d(...): Atrous convolution (a.k.a. convolution with holes or dilated convolution).
atrous_conv2d_transpose(...): The transpose of atrous_conv2d.

avg_pool(...): Performs the avg pooling on the input.
avg_pool1d(...): Performs the average pooling on the input.
avg_pool2d(...): Performs the average pooling on the input.
avg_pool3d(...): Performs the average pooling on the input.

batch_norm_with_global_normalization(...): Batch normalization.
batch_normalization(...): Batch normalization.

bias_add(...): Adds bias to value.


conv1d(...): Computes a 1-D convolution given 3-D input and filter tensors.
conv1d_transpose(...): The transpose of conv1d.
conv2d(...): Computes a 2-D convolution given 4-D input and filters tensors.
conv2d_transpose(...): The transpose of conv2d.
conv3d(...): Computes a 3-D convolution given 5-D input and filters tensors.
conv3d_transpose(...): The transpose of conv3d.
conv_transpose(...): The transpose of convolution.

convolution(...): Computes sums of N-D convolutions (actually cross-correlation).

ctc_beam_search_decoder(...): Performs beam search decoding on the logits given in input.

ctc_greedy_decoder(...): Performs greedy decoding on the logits given in input (best path).

ctc_loss(...): Computes CTC (Connectionist Temporal Classification) loss.

ctc_unique_labels(...): Get unique labels and indices for batched labels for tf.nn.ctc_loss.

depth_to_space(...): DepthToSpace for tensors of type T.

depthwise_conv2d(...): Depthwise 2-D convolution.
depthwise_conv2d_backprop_filter(...): Computes the gradients of depthwise convolution with respect to the filter.

depthwise_conv2d_backprop_input(...): Computes the gradients of depthwise convolution with respect to the input.

dilation2d(...): Computes the grayscale dilation of 4-D input and 3-D filters tensors.

dropout(...): Computes dropout.

embedding_lookup(...): Looks up ids in a list of embedding tensors.

embedding_lookup_sparse(...): Computes embeddings for the given ids and weights.

erosion2d(...): Computes the grayscale erosion of 4-D value and 3-D filters tensors.

fixed_unigram_candidate_sampler(...): Samples a set of classes using the provided (fixed) base distribution.

fractional_avg_pool(...): Performs fractional average pooling on the input.
fractional_max_pool(...): Performs fractional max pooling on the input.

in_top_k(...): Says whether the targets are in the top K predictions.

l2_loss(...): L2 Loss.

l2_normalize(...): Normalizes along dimension axis using an L2 norm.

learned_unigram_candidate_sampler(...): Samples a set of classes from a distribution learned during training.

local_response_normalization(...): Local Response Normalization.

log_poisson_loss(...): Computes log Poisson loss given log_input.



lrn(...): Local Response Normalization.

max_pool(...): Performs the max pooling on the input.
max_pool1d(...): Performs the max pooling on the input.
max_pool2d(...): Performs the max pooling on the input.
max_pool3d(...): Performs the max pooling on the input.
max_pool_with_argmax(...): Performs max pooling on the input and outputs both max values and indices.

moments(...): Calculates the mean and variance of x.

nce_loss(...): Computes and returns the noise-contrastive estimation training loss.

normalize_moments(...): Calculate the mean and variance of based on the sufficient statistics.

pool(...): Performs an N-D pooling operation.


safe_embedding_lookup_sparse(...): Lookup embedding results, accounting for invalid IDs and empty features.

sampled_softmax_loss(...): Computes and returns the sampled softmax training loss.

scale_regularization_loss(...): Scales the sum of the given regularization losses by number of replicas.

selu(...): Computes scaled exponential linear: scale * alpha * (exp(features) - 1)

separable_conv2d(...): 2-D convolution with separable filters.

sigmoid(...): Computes sigmoid of x element-wise.

sigmoid_cross_entropy_with_logits(...): Computes sigmoid cross entropy given logits.

softmax(...): Computes softmax activations.

softmax_cross_entropy_with_logits(...): Computes softmax cross entropy between logits and labels.

softplus(...): Computes softplus: log(exp(features) + 1).

softsign(...): Computes softsign: features / (abs(features) + 1).

space_to_batch(...): SpaceToBatch for N-D tensors of type T.

space_to_depth(...): SpaceToDepth for tensors of type T.

sparse_softmax_cross_entropy_with_logits(...): Computes sparse softmax cross entropy between logits and labels.

sufficient_statistics(...): Calculate the sufficient statistics for the mean and variance of x.


top_k(...): Finds values and indices of the k largest entries for the last dimension.

weighted_cross_entropy_with_logits(...): Computes a weighted cross entropy.

weighted_moments(...): Returns the frequency-weighted mean and variance of x.

with_space_to_batch(...): Performs op on the space-to-batch representation of input.

zero_fraction(...): Returns the fraction of zeros in value.
```

```


```

#
```
tf.nn.conv1d

```

```


```

```


```


#
```
https://www.tensorflow.org/api_docs/python/tf/nn/conv1d

tf.nn.conv1d(
    input,
    filters,
    stride,
    padding, ==== 'SAME' or 'VALID'兩種
    data_format='NWC',
    dilations=None,
    name=None
)
# Returns:A Tensor. Has the same type as input.


# 資料格式data_format
"NWC"資料格式 [batch, in_width, in_channels] 
"NCW"資料格式[batch, in_channels, in_width] 

a filter / kernel tensor of shape [filter_width, in_channels, out_channels],
 this op reshapes the arguments to pass them to conv2d to perform the equivalent convolution operation.

```
###  "deconvolution" 
```
after Deconvolutional Networks, 
but is really the transpose (gradient) of conv1d rather than an actual deconvolution.
```
```
tf.nn.conv1d_transpose(
    input,
    filters,
    output_shape,
    strides,
    padding='SAME',
    data_format='NWC',
    dilations=None,
    name=None
)
```

```


```


#
```


```

```


```

```


```


#
```


```

```


```

```


```

