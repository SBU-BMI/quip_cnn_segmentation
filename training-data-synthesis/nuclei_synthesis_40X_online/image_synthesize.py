import numpy as np
import pickle
import png
from PIL import Image, ImageDraw
from polygon import generatePolygon
from mask2image_otsu import Mask2Image
from scipy import ndimage
from sys import stdout

# Synthesize patches of 460x460 pixels first,
# then cut the surrounding 30 pixels off to get patches of 400x400 pixels.
size = 460;
cm = 30;

# Apply a QUAD transform with quad_marg.
# This transform models the correlation of shape and size of nearby nuclei.
quad_marg = 310;

# Initialize the Mask2Image object, using textures in real tiles.
m2image = Mask2Image('./nuclei_synthesis_40X_online/real_tiles/');

# This function returns a set of parameters for nuclear polygon generation.
def get_rand_polygon_param(s):
    x, y = int(np.random.rand()*s), int(np.random.rand()*s);
    rad = np.random.rand()*4.5 + 8.5;
    irr = 0.6+np.random.rand()*2.0;
    spike = 0.04+np.random.rand()*0.20;
    nverti = int(np.random.rand()*8+10);
    return x, y, rad, irr, spike, nverti;

# This function returns a set of parameters for noise polygon generation.
def get_rand_noise(s):
    x, y = int(np.random.rand()*s), int(np.random.rand()*s);
    rad = np.random.rand()*5.0+6.0;
    irr = 40.0;
    spike = 40.0;
    nverti = 40;
    return x, y, rad, irr, spike, nverti;

# Draw polygon mask according to polygon parameters
def draw_polygon(x, y, rad, irr, spike, nverti, s):
    vertices = generatePolygon(x, y, rad, irr, spike, nverti);
    mask = Image.fromarray(np.zeros((s, s, 3), dtype=np.uint8));
    draw = ImageDraw.Draw(mask);
    draw.polygon(vertices, fill=(1,1,1));
    return np.array(mask);

# Apply a random QUAD transformation.
def random_transform(mask, s, q):
    return np.array(Image.fromarray(mask).transform((s, s), Image.QUAD, q, Image.NEAREST));

def rand_quad_trans():
    q = (np.random.rand()*quad_marg,
            np.random.rand()*quad_marg,
            np.random.rand()*quad_marg,
            size+quad_marg-np.random.rand()*quad_marg,
            size+quad_marg-np.random.rand()*quad_marg,
            size+quad_marg-np.random.rand()*quad_marg,
            size+quad_marg-np.random.rand()*quad_marg,
            np.random.rand()*quad_marg);
    return q;

def rand_nucleus(nsize, qm, q, nsize_bias, cyto_mask):
    mask = np.array([-1]);
    ntry = 0;
    while ((mask>0).sum() <= 140 or ((mask*(1-cyto_mask)).sum() > 0)) and (ntry < 500):
        x, y, rad, irr, spike, nverti = get_rand_polygon_param(nsize+qm);
        mask = draw_polygon(x, y, int(nsize_bias*rad), irr, spike, nverti, nsize+qm);
        mask = random_transform(mask, nsize, q);
        ntry += 1;
    detect = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8);
    x, y, _ = np.nonzero(mask);
    detect[(np.min(x)+np.max(x))//2, (np.min(y)+np.max(y))//2, :] = 255;

    dx = ndimage.sobel(mask[:,:,0], 0);
    dy = ndimage.sobel(mask[:,:,0], 1);
    contour = ((dx>0) + (dy>0))[:,:,np.newaxis] * (mask==0);

    return mask, detect, contour;

def get_noise_mask(nsize, qm, star_noise):
    mask = np.array([-1]);
    ntry = 0;
    quad = rand_quad_trans();
    while np.sum(mask > 0) <= 10 and (ntry < 100):
        if star_noise:
            x, y, rad, irr, spike, nverti = get_rand_noise(nsize+qm);
        else:
            x, y, rad, irr, spike, nverti = get_rand_polygon_param(nsize+qm);
        mask = draw_polygon(x, y, rad, irr, spike, nverti, nsize+qm);
        mask = random_transform(mask, nsize, quad);
        ntry += 1;
    return mask;

def get_new_fake_image():
    dense_lvl, textures, full_tile_path = m2image.draw_random_texture(size, size);
    cyto_mask = m2image.get_cyto_mask(textures[1])

    if np.random.rand() < 0.10 and dense_lvl <= 1:
        nuclei_num = 0;
    else:
        nuclei_num = np.random.randn()*40 + dense_lvl*30 + 20.0;
    if nuclei_num < 0:
        nuclei_num = 0;
    if nuclei_num > 256:
        nuclei_num = 256;
    nuclei_num = int(nuclei_num);
    noise_num = np.random.randint(low=0, high=100);
    noise2_num = np.random.randint(low=0, high=5);
    nsize_bias = np.random.randn()*0.32 + dense_lvl/8.0 + 1.2;

    # mask is accumulative mask
    mask = np.zeros((size, size, 3), dtype=np.uint32);
    detect = np.zeros((size, size, 3), dtype=np.uint32);
    contour = np.zeros((size, size, 3), dtype=np.uint32);
    quad = rand_quad_trans();
    average_size = 0.0;
    for nuclei_no in range(nuclei_num):
        # return a mask to add on
        im_add, det_add, con_add = rand_nucleus(size, quad_marg, quad, nsize_bias, cyto_mask);
        overlap = np.sum((im_add * mask)>0) / np.sum(im_add>0).astype(np.float32);
        ntry = 0;
        # if the overlapping of nuclei is too large, return a new mask
        while (overlap > 0.15) and (ntry < 50):
            im_add, det_add, con_add = rand_nucleus(size, quad_marg, quad, nsize_bias, cyto_mask);
            overlap = np.sum((im_add * mask)>0) / np.sum(im_add>0).astype(np.float32);
            ntry += 1;
        mask += im_add;
        detect += det_add;
        contour += con_add;
        average_size += np.sum(mask[:,:,0]>0);
    if nuclei_num > 0:
        average_size /= nuclei_num;

    noise_mask = np.zeros((size, size, 3), dtype=np.uint8);
    for noise_no in range(noise_num):
        noise_mask += get_noise_mask(size, quad_marg, star_noise=True);

    noise2_mask = np.zeros((size, size, 3), dtype=np.uint8);
    for noise2_no in range(noise2_num):
        noise2_mask += get_noise_mask(size, quad_marg, star_noise=False);

    # convert mask to image and save the image
    image, refer, source, nucl, cyto, intp_nucl_mask = \
            m2image.go((mask>0).astype(np.uint8),
                    (noise_mask>0).astype(np.uint8), (noise2_mask>0).astype(np.uint8),
                    textures, average_size);
    #if np.random.rand() < 0.5:
    #    refer = np.transpose(refer, (1, 0, 2));
    #if np.random.rand() < 0.5:
    #    refer = refer[::-1, :, :];
    #if np.random.rand() < 0.5:
    #    refer = refer[:, ::-1, :];

    return image[cm:-cm, cm:-cm, :], mask[cm:-cm, cm:-cm, :], detect[cm:-cm, cm:-cm, :], \
           contour[cm:-cm, cm:-cm, :], refer[cm:-cm, cm:-cm, :], \
           source[cm:-cm, cm:-cm, :], nucl[cm:-cm, cm:-cm, :], \
           cyto[cm:-cm, cm:-cm, :], intp_nucl_mask[cm:-cm, cm:-cm, :], full_tile_path;

