import numpy as np

class Buffer(object):
  def __init__(self, config, rng):
    self.rng = rng
    self.buffer_size = config.buffer_size
    self.batch_size = config.batch_size

    image_dims = [
        config.input_height,
        config.input_width,
        config.input_channel,
    ]

    image_dims_grayscale = [
        config.input_height,
        config.input_width,
        1,
    ]

    self.idx = 0
    self.data = np.zeros([self.buffer_size,] + image_dims, dtype=np.float32)
    self.mask_data = np.zeros([self.buffer_size,] + image_dims_grayscale, dtype=np.float32)
    self.ref_data = np.zeros([self.buffer_size,] + image_dims, dtype=np.float32)

  def push(self, both_batches):
    batch = both_batches[0]
    mask = both_batches[1]
    ref_image = both_batches[2]
    batch_size = len(batch)
    if self.idx + batch_size > self.buffer_size:
      random_idx1 = self.rng.choice(self.idx, self.batch_size/2)
      random_idx2 = self.rng.choice(batch_size, self.batch_size/2)
      self.data[random_idx1] = batch[random_idx2]
      self.mask_data[random_idx1] = mask[random_idx2]
      self.ref_data[random_idx1] = ref_image[random_idx2]
    else:
      self.data[self.idx:self.idx+batch_size] = batch
      self.mask_data[self.idx:self.idx+batch_size] = mask
      self.ref_data[self.idx:self.idx+batch_size] = ref_image
      self.idx += batch_size


  def sample(self, n=None):
    assert self.idx > n, "not enough data is pushed"
    if n is None:
      n = self.batch_size/2
    random_idx = self.rng.choice(self.idx, n)

    return self.data[random_idx], self.mask_data[random_idx], self.ref_data[random_idx];


