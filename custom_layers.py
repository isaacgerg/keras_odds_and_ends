import tensorflow as tf

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

