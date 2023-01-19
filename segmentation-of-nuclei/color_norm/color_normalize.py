from . import stainNorm_Reinhard
from utils import imread

class reinhard_normalizer():
    def __init__(self, target_file):
        target_40X = imread(target_file)
        self.n_40X = stainNorm_Reinhard.Normalizer()
        self.n_40X.fit(target_40X)

    def normalize(self, image):
        # image RGB in uint8
        return self.n_40X.transform(image)

