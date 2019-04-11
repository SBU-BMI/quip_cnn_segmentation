# The refiner takes an image patch and attempts to make it more "realistic"

#Requires params dict to contain (resnet_refiner, with_ref, input_height, input_width, bn, refiner_dense_bias)
def refiner_model_fn(features, labels, mode, params):
#  def _build_refiner(self, synthesized, rlyer):
    with tf.variable_scope("refiner") as sc:
      if params['resnet_refiner']:
        layer = conv2d(features['synthesized'], 32, 3, 1, padding='SAME', normalizer_fn=params['bn'], scope="pre_conv_0");

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
          rlyer = tf.tile(rlyer, [1, params['input_height'], params['input_width'], 1]);
          layer = tf.concat([layer, rlyer], 3)

        layer = repeat(layer, 5, resnet_block, padding='SAME', normalizer_fn=params['bn'], scope="resnet")
        layer = conv2d(layer, 3, 3, 1, padding='SAME', activation_fn=identity, scope="conv_1");

      else:
        layer = conv2d(features['synthesized'], 32, 3, 1, padding='VALID', normalizer_fn=params['bn'], scope="pre_conv_0");

        if params['with_ref']:
          rlyer = conv2d(rlyer, 32, 5, 1, padding='VALID', normalizer_fn=params['bn'], scope="ref_conv_0");
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

        layer = conv2d(layer, 48, 5, 1, padding='VALID', normalizer_fn=params['bn'], scope="conv_0")
        layer = conv2d(layer, 64, 5, 1, padding='VALID', normalizer_fn=params['bn'], scope="conv_1")
        layer = slim.avg_pool2d(layer, 3, 2, scope="pool_1")
        layer = conv2d(layer, 80, 3, 1, padding='VALID', normalizer_fn=params['bn'], scope="conv_2")
        layer = conv2d(layer, 80, 3, 1, padding='VALID', normalizer_fn=params['bn'], scope="conv_3")
        layer = conv2d(layer, 80, 3, 1, padding='VALID', normalizer_fn=params['bn'], scope="conv_4")
        layer = conv2d(layer, 80, 3, 1, padding='VALID', normalizer_fn=params['bn'], scope="conv_5")
        layer = conv2d(layer, 96, 3, 1, padding='VALID', normalizer_fn=params['bn'], scope="conv_6")
        layer = conv2d(layer, 96, 3, 1, padding='VALID', normalizer_fn=params['bn'], scope="conv_7")
        layer = conv2d(layer, 96, 3, 1, padding='VALID', normalizer_fn=params['bn'], scope="conv_8")
        layer = conv2d(layer, 96, 3, 1, padding='VALID', normalizer_fn=params['bn'], scope="conv_9")
        layer = conv2d(layer, 104,3, 1, padding='VALID', normalizer_fn=params['bn'], scope="conv_10")
        layer = conv2d(layer, 104,3, 1, padding='VALID', normalizer_fn=params['bn'], scope="conv_11")
        layer = conv2d(layer, 32, 3, 1, padding='VALID', normalizer_fn=params['bn'], scope="conv_12")
        layer = tf.contrib.layers.flatten(layer)
        if params['refiner_dense_bias']:
          layer = dense(layer, 8*(params['input_height']+4)*(params['input_width']+4),
                  normalizer_fn=params['bn'], scope="dense")
        else:
          layer = dense(layer, 8*(params['input_height']+4)*(params['input_width']+4),
                  biases_initializer=None, normalizer_fn=params['bn'], scope="dense")
        layer = tf.reshape(layer, [-1, (params['input_height']+4), (params['input_width']+4), 8]);
        layer = conv2d(layer, 16, 3, 1, padding='VALID', normalizer_fn=params['bn'], scope="conv_13")
        layer = conv2d(layer, 3,  3, 1, padding='VALID', normalizer_fn=None, activation_fn=identity, scope="conv_14")


      output = tf.clip_by_value(synthesized+layer,
               clip_value_min=-1.0, clip_value_max=1.0, name="refiner_output_clip")
      #self.refiner_vars = tf.contrib.framework.get_variables(sc)
    return output

