import numpy as np
import pickle
import png
from PIL import Image, ImageDraw
from polygon import generatePolygon
from mask2image import Mask2Image
from scipy import ndimage
from sys import stdout

size = 200;
quad_marg = 120;
cm = 30;
mask2image = Mask2Image('./nuclei_synthesis_40X_online/real_tiles/');

def get_rand_polygon_param(s):
    x, y = int(np.random.rand()*s), int(np.random.rand()*s);
    rad = np.random.rand()*2.0 + 8.0;
    irr = 0.6+np.random.rand()*0.6;
    spike = 0.04+np.random.rand()*0.08;
    nverti = int(np.random.rand()*8+10);
    return x, y, rad, irr, spike, nverti;

def get_rand_noise(s):
    x, y = int(np.random.rand()*s), int(np.random.rand()*s);
    rad = np.random.rand()*4.0+4.0;
    irr = 40.0;
    spike = 40.0;
    nverti = 40;
    return x, y, rad, irr, spike, nverti;

def draw_polygon(x, y, rad, irr, spike, nverti, s):
    vertices = generatePolygon(x, y, rad, irr, spike, nverti);
    mask = Image.fromarray(np.zeros((s, s, 3), dtype=np.uint8));
    draw = ImageDraw.Draw(mask);
    draw.polygon(vertices, fill=(1,1,1));
    return np.array(mask);

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

def rand_nucleus(nsize, qm, q, nsize_bias):
    mask = np.array([-1]);
    ntry = 0;
    while np.sum(mask > 0) <= 140 and (ntry < 500):
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
    dense_lvl, textures, full_tile_path = mask2image.draw_random_texture(size, size);

    if np.random.rand() < 0.15:
        nuclei_num = 0;
    else:
        nuclei_num = np.random.randn()*8 + dense_lvl*3 + 10.0;
    if nuclei_num < 0:
        nuclei_num = 0;
    if nuclei_num > 36:
        nuclei_num = 36;
    nuclei_num = int(nuclei_num);
    noise_num = np.random.randint(low=0, high=7);
    noise2_num = np.random.randint(low=0, high=15);
    nsize_bias = np.random.randn()*0.3 + dense_lvl/12.0 + 1.2;

    # mask is accumulative mask
    mask = np.zeros((size, size, 3), dtype=np.uint32);
    detect = np.zeros((size, size, 3), dtype=np.uint32);
    contour = np.zeros((size, size, 3), dtype=np.uint32);
    quad = rand_quad_trans();
    average_size = 0.0;
    for nuclei_no in range(nuclei_num):
        # return a mask to add on
        im_add, det_add, con_add = rand_nucleus(size, quad_marg, quad, nsize_bias);
        best_im_add, best_det_add, best_con_add = im_add, det_add, con_add;
        best_overlap = np.sum((im_add * mask)>0);
        ntry = 0;
        # if the overlapping of nuclei is too large, return a new mask
        # try this for several times and pick the least overlapping new mask
        while (best_overlap > 0) and (ntry < 50):
            im_add, det_add, con_add = rand_nucleus(size, quad_marg, quad, nsize_bias);
            overlap = np.sum((im_add * mask)>0);
            if overlap < best_overlap:
                best_overlap = overlap;
                best_im_add, best_det_add, best_con_add = im_add, det_add, con_add;
            ntry += 1;
        mask += best_im_add;
        detect += best_det_add;
        contour += best_con_add;
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
    image, refer, nucl, cyto, intp_nucl_mask = \
            mask2image.go((mask>0).astype(np.uint8),
                    (noise_mask>0).astype(np.uint8), (noise2_mask>0).astype(np.uint8),
                    textures, average_size);
    if np.random.rand() < 0.5:
        refer = np.transpose(refer, (1, 0, 2));
    if np.random.rand() < 0.5:
        refer = refer[::-1, :, :];
    if np.random.rand() < 0.5:
        refer = refer[:, ::-1, :];

    return image[cm:-cm, cm:-cm, :], mask[cm:-cm, cm:-cm, :], detect[cm:-cm, cm:-cm, :], \
           contour[cm:-cm, cm:-cm, :], refer[cm:-cm, cm:-cm, :], nucl[cm:-cm, cm:-cm, :], \
           cyto[cm:-cm, cm:-cm, :], intp_nucl_mask[cm:-cm, cm:-cm, :], full_tile_path;

