#The discriminator takes image patches as input and attempts to determine whether they are real or synthetic.


#Currently requires params dict to contain (reuse, name, with_ref, bn, input_width, input_height)
def discrim_model_fn(features, labels, mode, params):
#  def _build_discrim(self, layer, rlyer, name, reuse=False):
    with tf.variable_scope("discriminator", reuse=params['reuse']) as sc:
      layer = conv2d(features['input_image'], 32, 3, 1, padding='VALID', normalizer_fn=params['bn'], scope="conv_0", name=params['name'])

      if params['with_ref']:
        rlyer = conv2d(features['rlyer'], 32, 5, 1, padding='VALID', normalizer_fn=params['bn'], scope="ref_conv_0");
        rlyer = slim.avg_pool2d(rlyer, 3, 2, scope="ref_pool_0")
        rlyer = conv2d(rlyer, 32, 5, 1, padding='VALID', normalizer_fn=params['bn'], scope="ref_conv_1");
        rlyer = conv2d(rlyer, 32, 5, 1, padding='VALID', normalizer_fn=params['bn'], scope="ref_conv_2");
        rlyer = conv2d(rlyer, 32, 5, 1, padding='VALID', normalizer_fn=params['bn'], scope="ref_conv_3");
        rlyer = conv2d(rlyer, 32, 5, 1, padding='VALID', normalizer_fn=params['bn'], scope="ref_conv_4");
        pool_height = (params['input_height']-4-1)/2-4-4-4-4;
        pool_width = (params['input_width']-4-1)/2-4-4-4-4;
        rlyer = slim.avg_pool2d(rlyer, (pool_height, pool_width), scope="ref_pool_1");
        rlyer = tf.tile(rlyer, [1, params['input_height']-2, params['input_width']-2, 1]);
        layer = tf.concat([layer, rlyer], 3)

      layer = conv2d(layer, 48, 3, 1, padding='VALID', normalizer_fn=params['bn'], scope="conv_1", name=params['name'])
      layer = slim.avg_pool2d(layer, 3, 2, scope="pool_1")
      layer = conv2d(layer, 64, 5, 1, padding='VALID', normalizer_fn=params['bn'], scope="conv_2", name=params['name'])
      layer = conv2d(layer, 80, 5, 1, padding='VALID', normalizer_fn=params['bn'], scope="conv_3", name=params['name'])
      layer = conv2d(layer, 80, 3, 1, padding='VALID', normalizer_fn=params['bn'], scope="conv_4", name=params['name'])
      layer = slim.avg_pool2d(layer, 3, 2, scope="pool_2")
      layer = conv2d(layer, 80, 3, 1, padding='VALID', normalizer_fn=params['bn'], scope="conv_5", name=params['name'])
      layer = conv2d(layer, 80, 3, 1, padding='VALID', normalizer_fn=params['bn'], scope="conv_6", name=params['name'])
      layer = conv2d(layer, 80, 3, 1, padding='VALID', normalizer_fn=params['bn'], scope="conv_7", name=params['name'])
      layer = conv2d(layer, 64, 1, 1, padding='VALID', normalizer_fn=params['bn'], scope="conv_8", name=params['name'])
      logits = conv2d(layer, 2, 1, 1, padding='VALID', normalizer_fn=None, activation_fn=identity, scope="conv_9", name=params['name'])
      output = tf.nn.softmax(logits, name="softmax")
      #self.discrim_vars = tf.contrib.framework.get_variables(sc)
#    return output, logits

      predictions = {
        "output": output,
        "logits": logits
      }

      if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

      #Loss

      #Correct rv for other modes

