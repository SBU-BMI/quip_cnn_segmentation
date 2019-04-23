import numpy as np
import cv2
from scipy import misc
from PIL import Image
from skimage.color import hed2rgb, rgb2hed
from os import listdir
from sys import stdout
from os.path import isfile, join
from my_canny import canny_edge_on_mask
from scipy import ndimage
from scipy import stats

class Mask2Image:
    def __init__(self, tile_path):
        self.tile_path = tile_path;
        self.paths = [f for f in listdir(tile_path) if isfile(join(tile_path, f))];

    def rand_blur_mask(self, mask, radius):
        filt = np.random.random((radius, radius)) + 5;
        filt = (filt / np.sum(filt)).astype(np.float32);
        mask = ndimage.convolve(mask, filt);
        return mask;

    def gaussian_blur(self, im, sigma):
        return ndimage.filters.gaussian_filter(im, sigma, mode='mirror')

    def rand_blur_and_change_intensity(self, im):
        if np.random.rand() < 0.5:
            if np.random.rand() < 0.7:
                radius = 3;
            else:
                radius = 5;
            filt = np.random.random((radius, radius)) + 3;
            filt[(radius-1)/2, (radius-1)/2] += np.random.rand()*10;
            filt = (filt / np.sum(filt)).astype(np.float32);
            im = ndimage.convolve(im, filt);

        a = np.random.random((int(np.random.rand()*7)+2, int(np.random.rand()*7)+2));
        b = np.array(Image.fromarray((a*255).astype(np.uint8)).resize((im.shape[0], im.shape[1])));
        c = b.astype(np.float32) / 255.0;
        d = 0.66*(c-0.5) + 1.0;
        e = self.rand_blur_mask(d, im.shape[0]//5);

        return im*e;

    def aug_color(self, img, mag):
        adj_range = 0.09 * mag;
        adj_add = 6 * mag;
        rgb_mean = np.mean(img.astype(np.float32), axis=(1,2), keepdims=True);
        adj_magn = np.random.uniform(1-adj_range, 1+adj_range, (1, 1, 3)).astype(np.float32);
        img = np.clip((img.astype(np.float32)-rgb_mean)*adj_magn + rgb_mean +
                np.random.uniform(-1.0, 1.0, (1, 1, 3))*adj_add, 0.1, 249.9);
        return img;

    def add_h_color(self, img):
        hed = rgb2hed(img / 255.0)
        hed[..., 0] += np.random.rand() * 0.04
        img = hed2rgb(hed)
        img = np.clip(img * 255, 0, 255)
        return img;

    def interpolate_mask(self, mask, radius):
        filt = np.random.random((radius, radius)) + 2;
        filt[(radius-1)/2, (radius-1)/2] += (2 + np.random.rand()*12);
        filt = (filt / np.sum(filt)).astype(np.float32);
        im1 = ndimage.convolve(mask[:, :, 0], filt);
        im2 = ndimage.convolve(mask[:, :, 1], filt);
        im3 = ndimage.convolve(mask[:, :, 2], filt);
        overall = np.stack([im1, im2, im3], axis=2);

        if np.random.rand() < 0.85:
            p_iter_n = 2;
            for patial_iter in range(p_iter_n):
                radius = 5;
                marg = (radius-1)//2;
                size0, size1 = mask.shape[0], mask.shape[1];
                x0, y0 = np.random.randint(0, size0//2-marg-1), np.random.randint(0, size1//2-marg-1);
                x1, y1 = np.random.randint(size0//2+marg+1, size0), np.random.randint(size1//2+marg+1, size1);
                filt = np.random.random((radius, radius)) + 2;
                filt[(radius-1)/2, (radius-1)/2] += (16 + np.random.rand()*48);
                filt = (filt / np.sum(filt)).astype(np.float32);
                im1 = ndimage.convolve(overall[x0:x1, y0:y1, 0], filt);
                im2 = ndimage.convolve(overall[x0:x1, y0:y1, 1], filt);
                im3 = ndimage.convolve(overall[x0:x1, y0:y1, 2], filt);
                patial = np.stack([im1, im2, im3], axis=2);
                overall[x0+marg:x1-marg, y0+marg:y1-marg, :] = patial[marg:-marg, marg:-marg, :];

        return overall;

    def draw_random_texture_try(self, only_want_nuc_texture, size0, size1):
        pathid = int(np.random.rand()*len(self.paths));
        full_tile_path = self.paths[pathid];

        full_tile = np.array(Image.open(join(self.tile_path, full_tile_path)).convert('RGB'));
        hed = rgb2hed((full_tile.astype(np.float32)/255.0).astype(np.float32));

        nuc_texture = 1.3 * (hed[:,:,1] - np.min(hed[:,:,1])) / (np.max(hed[:,:,1]) - np.min(hed[:,:,1]))
        x = int(np.random.rand()*(nuc_texture.shape[0]-size0));
        y = int(np.random.rand()*(nuc_texture.shape[1]-size1));
        nuc_texture = nuc_texture[x:x+size0, y:y+size1].copy();

        if only_want_nuc_texture:
            return True, True, True, True, True, True, True, nuc_texture, True;

        hed_mean = np.mean(hed[:,:,0]);
        if hed_mean < -1.33 or hed_mean > -1.1:
            return False, False, False, False, False, False, False, False, False;

        gray = (255 * (hed[:,:,0] - hed[:,:,0].min()) / (hed[:,:,0].max() - hed[:,:,0].min())).astype(np.uint8);
        nuc_color_thr, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU);
        nuc_color_thr_per = stats.percentileofscore(gray.flatten(), nuc_color_thr)
        nuc_color_per = 100 - (100 - nuc_color_thr_per) * 0.92
        nuc_map_per = 100 - (100 - nuc_color_thr_per) * 1.08
        dense_lvl = (thresholded > 0).sum() / 0.06 / thresholded.size

        to_map_mask = (hed[:,:,0] > np.percentile(hed[:,:,0], nuc_map_per));
        to_map_mask = (self.gaussian_blur(to_map_mask.astype(np.float32), 0.5) > 0.50);

        nucl_color_mask = (hed[:,:,0] > np.percentile(hed[:,:,0], nuc_color_per)) * \
                          (hed[:,:,0] < np.percentile(hed[:,:,0], 99));
        nucl_color = [np.mean(full_tile[:,:,0][nucl_color_mask]), \
                      np.mean(full_tile[:,:,1][nucl_color_mask]), \
                      np.mean(full_tile[:,:,2][nucl_color_mask])];
        noise_color_mask = (hed[:,:,0] > np.percentile(hed[:,:,0],
                                (nuc_color_per+35)//2 - np.random.randint(0,3) - 25)) * \
                           (hed[:,:,0] < np.percentile(hed[:,:,0],
                                (nuc_color_per+35)//2 - np.random.randint(0,3) - 12));
        noise_color = [np.mean(full_tile[:,:,0][noise_color_mask]), \
                       np.mean(full_tile[:,:,1][noise_color_mask]), \
                       np.mean(full_tile[:,:,2][noise_color_mask])];
        if np.min(nucl_color) > 100:
            return False, False, False, False, False, False, False, False, False;

        x, y, fx, fy = self.sample_xy_fxfy(full_tile.shape[0]-size0, full_tile.shape[1]-size1, size0, size1);
        cyto_tile = cv2.inpaint(full_tile[x:x+size0, y:y+size1, :], \
                to_map_mask.astype(np.uint8)[x:x+size0, y:y+size1], 3, cv2.INPAINT_TELEA);

        source_tl = full_tile[x:x+size0, y:y+size1, :].copy();
        full_tile = full_tile[fx:fx+size0, fy:fy+size1, :];
        return True, dense_lvl, full_tile, cyto_tile, source_tl, nucl_color, noise_color, nuc_texture, full_tile_path;

    def draw_random_texture(self, size0, size1):
        success = False;
        while not success:
            success, dense_lvl, full_tile, cyto_tile, source_tl, nucl_color, noise_color, nuc_texture1, full_tile_path = \
                    self.draw_random_texture_try(False, size0, size1);
        success = False;
        while not success:
            success, dense_lvl, _, _, _, _, _, nuc_texture2, _ = \
                    self.draw_random_texture_try(True, size0, size1);
        return dense_lvl, (full_tile, cyto_tile, source_tl, nucl_color, noise_color, nuc_texture1, nuc_texture2), full_tile_path;

    def sample_overlap(self, x, y, fx, fy, xlen, ylen):
        if fx <= x and x <= fx+xlen and fy <= y and y <= fy+ylen:
            return True;
        if fx <= x+xlen and x+xlen <= fx+xlen and fy <= y and y <= fy+ylen:
            return True;
        if fx <= x and x <= fx+xlen and fy <= y+ylen and y+ylen <= fy+ylen:
            return True;
        if fx <= x+xlen and x+xlen <= fx+xlen and fy <= y+ylen and y+ylen <= fy+ylen:
            return True;
        return False;

    def sample_xy_fxfy(self, size0, size1, xlen, ylen):
        x, y, fx, fy = 0, 0, 0, 0;
        while self.sample_overlap(x, y, fx, fy, xlen, ylen):
            x, y = int(np.random.rand()*size0), int(np.random.rand()*size1);
            fx, fy = int(np.random.rand()*size0), int(np.random.rand()*size1);
        return x, y, fx, fy;

    def float2image(self, float_mat):
        return np.round(np.clip(float_mat, 0.1, 249.9)).astype(np.uint8);

    def mask_to_mixing_mask(self, mask, nuc_texture_transpar, average_nucl_size, more_blur):
        salt_pep =  (0.0 + 0.06*np.random.rand()) * np.random.random((mask.shape[0]//2, mask.shape[1]//2));
        salt_pep = misc.imresize(salt_pep, (mask.shape[0], mask.shape[1])).astype(np.float32) / 255.0;
        salt_pep += (0.0 + 0.06*np.random.rand()) * np.random.random((mask.shape[0], mask.shape[1]));
        nuc_texture_transpar = (nuc_texture_transpar - np.min(nuc_texture_transpar)) \
                / (np.max(nuc_texture_transpar) - np.min(nuc_texture_transpar)) + salt_pep;

        if average_nucl_size < 800.0:
            transpar_per = np.random.rand() * 15;
        elif average_nucl_size > 2400.0:
            transpar_per = 10.0 + np.random.rand() * 42;
        elif np.random.rand() < 0.40:
            transpar_per = np.random.rand() * 15;
        else:
            transpar_per = 0.0 + np.random.rand() * 42;
        nuc_texture_transpar = (nuc_texture_transpar > np.percentile(nuc_texture_transpar, transpar_per)).astype(np.uint8);

        mask_edge = canny_edge_on_mask(((mask[:,:,0]>0).astype(np.uint8) * 255).astype(np.uint8));
        # Thicken mask edge
        if transpar_per > 10 and transpar_per < 30 and np.random.rand() < 0.30:
            mask_edge = mask_edge + (mask[:,:,0]>0) - \
                (self.rand_blur_mask((mask[:,:,0]>0).astype(np.float32), np.random.randint(low=5, high=17)) > 0.9999);

        # Drop random pixels on mask edge
        if transpar_per >= 20:
            mask_edge *= (np.random.random((mask.shape[0], mask.shape[1])) > 0.0+0.30*np.random.rand());

        mask_edge = np.repeat(mask_edge[..., np.newaxis], 3, axis=2);

        rr = np.random.rand();
        if more_blur:
            if rr < 0.33:
                mask_inter_radius = 7;
            else:
                mask_inter_radius = 9;
        elif transpar_per < 20:
            if rr < 0.05 or average_nucl_size < 700.0:
                mask_inter_radius = 3;
            elif rr < 0.70:
                mask_inter_radius = 5;
            else:
                mask_inter_radius = 7;
        else:
            if rr < 0.10:
                mask_inter_radius = 3;
            else:
                mask_inter_radius = 5;

        mixed_mask = mask * nuc_texture_transpar[..., np.newaxis] + mask_edge;
        recall_mask = (np.random.random((mask.shape[0]//5, mask.shape[1]//5))*255).astype(np.uint8);
        recall_mask = misc.imresize(recall_mask, (mask.shape[0], mask.shape[1]));
        recall_mask = (recall_mask > 150+100*np.random.rand());
        mixed_mask[:,:,0] += mask[:,:,0] * recall_mask;
        mixed_mask[:,:,1] += mask[:,:,1] * recall_mask;
        mixed_mask[:,:,2] += mask[:,:,2] * recall_mask;

        intp_nucl_mask = self.interpolate_mask((mixed_mask>0.5).astype(np.float32), mask_inter_radius);
        return intp_nucl_mask;

    def mask_to_mixing_mask_clutered(self, mask, nuc_texture_transpar, average_nucl_size):
        salt_pep = (0.20 + 0.30*np.random.rand()) * np.random.random((mask.shape[0], mask.shape[1]));
        nuc_texture_transpar = (nuc_texture_transpar - np.min(nuc_texture_transpar)) \
                / (np.max(nuc_texture_transpar) - np.min(nuc_texture_transpar)) + salt_pep;
        transpar_per = 85;
        nuc_texture_transpar = (nuc_texture_transpar > np.percentile(nuc_texture_transpar, transpar_per)).astype(np.uint8);
        mask = mask * nuc_texture_transpar[..., np.newaxis];
        rr = np.random.rand();
        if rr < 0.60:
            mask_inter_radius = 3;
        elif rr < 0.98:
            mask_inter_radius = 5;
        else:
            mask_inter_radius = 7;

        intp_nucl_mask = self.interpolate_mask((mask>0.5).astype(np.float32), mask_inter_radius);
        return intp_nucl_mask;

    def get_nucl_color_map(self, nuc_texture_back_ground, nucl_color):
        nucl = np.repeat(nuc_texture_back_ground[..., np.newaxis], 3, axis=2);
        nucl[:, :, 0] *= nucl_color[0];
        nucl[:, :, 1] *= nucl_color[1];
        nucl[:, :, 2] *= nucl_color[2];
        nucl[:, :, 0] += 3.2*(np.random.random((nucl.shape[0], nucl.shape[1]))-0.5);
        nucl[:, :, 1] += 2.6*(np.random.random((nucl.shape[0], nucl.shape[1]))-0.5);
        nucl[:, :, 2] += 5.2*(np.random.random((nucl.shape[0], nucl.shape[1]))-0.5);
        nucl[:, :, 0] = self.rand_blur_and_change_intensity(nucl[:, :, 0]);
        nucl[:, :, 1] = self.rand_blur_and_change_intensity(nucl[:, :, 1]);
        nucl[:, :, 2] = self.rand_blur_and_change_intensity(nucl[:, :, 2]);
        return nucl;

    def get_cyto_mask(self, cyto):
        step = cyto.shape[0]//13;
        tissue_mask = np.zeros((cyto.shape[0], cyto.shape[1]), dtype=np.float32);
        for x in range(0, len(cyto)-1, step):
            for y in range(0, len(cyto)-1, step):
                patlet = cyto[x:x+step, y:y+step, :];
                tissue_mask[x:x+step, y:y+step] = patlet.std(axis=(0,1)).mean();
        tissue_mask = ndimage.convolve(tissue_mask, np.ones((3,3), dtype=np.float32)/9.0);
        return (tissue_mask >= 8.0)[..., np.newaxis]

    def go(self, mask, noise_mask, noise2_mask, textures, average_nucl_size):
        full_tile, cyto_tile, source_tl, nucl_color, noise2_color, nuc_texture1, nuc_texture2 = textures;

        cyto = self.add_h_color(self.aug_color(cyto_tile, mag=0.10));

        nucl = self.get_nucl_color_map(nuc_texture1, nucl_color);
        nucl = self.aug_color(nucl, mag=0.10);
        intp_nucl_mask = self.mask_to_mixing_mask(mask, nuc_texture2.copy(), average_nucl_size, more_blur=False);

        ########################
        # Adding noise type2
        noise2 = self.get_nucl_color_map(nuc_texture1, noise2_color);
        noise2 = self.aug_color(noise2, mag=0.02);
        intp_noise2_mask = self.mask_to_mixing_mask(noise2_mask, nuc_texture2.copy(), average_nucl_size, more_blur=True);
        cyto = cyto * (1-intp_noise2_mask) + noise2 * intp_noise2_mask;
        # Adding noise type2
        ########################

        ########################
        # Adding noise type1
        noise = self.get_nucl_color_map(nuc_texture1, nucl_color);
        noise = self.aug_color(noise, mag=1.0);
        intp_noise_mask = self.mask_to_mixing_mask_clutered(noise_mask, nuc_texture2.copy(), average_nucl_size);
        cyto = cyto * (1-intp_noise_mask) + noise * intp_noise_mask;
        # Adding noise type1
        ########################

        final = cyto * (1-intp_nucl_mask) + nucl * intp_nucl_mask;

        return self.float2image(final), full_tile, source_tl, \
               self.float2image(nucl), self.float2image(cyto), \
               self.float2image(255*intp_nucl_mask);

