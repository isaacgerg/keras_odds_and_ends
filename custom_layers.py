import tensorflow as tf

#-----------------------------------------------------------------------------------------------------------------------
# Computes the norm channel-wise of all the vectors and does max-pooling based on it.
class ChannelNormPooling(tf.keras.layers.Layer):
    def __init__(self, pool_size=(2, 2), strides=None, padding='VALID', **kwargs):
        super(ChannelNormPooling, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides or pool_size
        self.padding = padding

    def call(self, inputs):
        # Compute channel-wise norms of the input tensor
        norms = tf.norm(inputs, axis=-1)[:,:,:,None]
        
        pool, indices = tf.nn.max_pool_with_argmax(norms, ksize=self.pool_size, strides=self.strides, padding=self.padding, include_batch_in_index=False)
        b,h_i,w_i, c = list(indices.shape)
        b,h,w,c = list(inputs.shape)
        
        indices = tf.reshape(indices, (-1, h_i*w_i, 1))
        iut_rs = tf.reshape(inputs, (-1, h * w , c))
            
        r = tf.gather_nd(iut_rs, indices, batch_dims=1)
        r = tf.reshape(r, [-1, h_i, w_i, c])

        return r

    def get_config(self):
        config = super(ChannelNormPooling, self).get_config()
        config.update({
            'pool_size': self.pool_size,
            'strides': self.strides,
            'padding': self.padding,
        })
        return config

#-----------------------------------------------------------------------------------------------------------------------
class whiten(tf.keras.layers.Layer):
    def build(self, input_shape):
        super(whiten, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        mn = K.mean(x)
        std = K.std(x) + K.epsilon()
        r = (x - mn) / std
        return r

    def compute_output_shape(self, input_shape):
        return input_shape

#-----------------------------------------------------------------------------------------------------------------------
class Bias(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        #self.output_dim = output_dim
        super(Bias, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.bias = self.add_weight(name='bias',
                                      shape=(input_shape[3],),
                                      initializer='glorot_normal',
                                      trainable=True)
        self.input_dim = input_shape
        super(Bias, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        r = K.bias_add(x, self.bias, 'channels_last')
        return r

    def compute_output_shape(self, input_shape):
        return input_shape
    #
#

#-----------------------------------------------------------------------------------------------------------------------
class fft2d(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(fft2d, self).__init__(**kwargs)

    def build(self, input_shape):
        assert(len(input_shape)==4)
        assert(input_shape[3]==1)
        self.input_dim = input_shape
        super(fft2d, self).build(input_shape)  

    def call(self, x):
        r = tf.squeeze(x, axis=3)
        r = tf.fft2d(r)   
        r = tf.expand_dims(r, axis=3) 
        return r

    def compute_output_shape(self, input_shape):
        return input_shape
    #
#
#-----------------------------------------------------------------------------------------------------------------------
class ifft2d(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ifft2d, self).__init__(**kwargs)

    def build(self, input_shape):
        assert(len(input_shape)==4)
        assert(input_shape[3]==1)
        self.input_dim = input_shape
        super(ifft2d, self).build(input_shape)  

    def call(self, x):
        r = tf.squeeze(x, axis=3)
        r = tf.ifft2d(r)   
        r = tf.expand_dims(r, axis=3) 
        return r

    def compute_output_shape(self, input_shape):
        return input_shape
    #
#
#-----------------------------------------------------------------------------------------------------------------------
class ifftshift2d(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ifftshift2d, self).__init__(**kwargs)

    def call(self, x):
        assert(tf.keras.backend.ndim(x) == 4)
        shiftX = x.shape[1].value
        shiftY = x.shape[2].value
        r = tf.keras.layers.Lambda(lambda x: tf.roll(x, shift=(-shiftX, -shiftY), axis=(1,2)))(x)
        return r

    def compute_output_shape(self, input_shape):
        return input_shape
    #
#
#-----------------------------------------------------------------------------------------------------------------------
class fftshift2d(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(fftshift2d, self).__init__(**kwargs)

    def call(self, x):
        assert(tf.keras.backend.ndim(x) == 4)
        shiftX = x.shape[1].value
        shiftY = x.shape[2].value
        r = tf.keras.layers.Lambda(lambda x: tf.roll(x, shift=(shiftX, shiftY), axis=(1,2)))(x)
        return r

    def compute_output_shape(self, input_shape):
        return input_shape
    #
#

#-----------------------------------------------------------------------------------------------------------------------
class schlick(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(schlick, self).__init__(**kwargs)    
    
    def normalize(self, x):
        x = x - tf.keras.backend.min(x, axis=1, keepdims=True)
        x = x / (tf.keras.backend.max(x, axis=1, keepdims=True) + tf.keras.backend.epsilon())
        return x    
    
    def call(self, x):
        tileSize = x.shape[1]
        x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.batch_flatten(x))(x)
        
        x = self.normalize(x)
        # assume x is batchSize, h*w
        m = tf.contrib.distributions.percentile(x, 50.0, axis=1, keep_dims=True)
        targetBrightness = 0.2
        b = (targetBrightness - targetBrightness * m) / (m - targetBrightness * m)
        b = tf.clip_by_value(b, 1.0, 99999999.0)
        # apply b
        L = tf.abs((b*x) / ((b-1)*x + 1 + tf.keras.backend.epsilon()))
        
        L = tf.keras.layers.Lambda(lambda x: tf.keras.backend.reshape(x, (-1, tileSize, tileSize, 1)))(L)
        
        return L
    
#-----------------------------------------------------------------------------------------------------------------------

