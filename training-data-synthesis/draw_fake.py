import numpy as np
import sys
from PIL import Image
from nuclei_synthesis_40X_online.image_synthesize import get_new_fake_image

im_no = int(sys.argv[1]);
im_no_inc = int(sys.argv[2]);
fid = open('output/fake_info_{}.txt'.format(im_no), 'w');

# only synthesize 25,000 patches
while im_no < 25000:
    try:
        image, mask, detect, contour, refer, source, nucl, cyto, intp_mask, tile_path = get_new_fake_image();
    except Exception as e:
        print(e);
        continue;
    mask = (mask>0).astype(np.uint8);
    detect = (detect>0).astype(np.uint8);
    contour = (contour>0).astype(np.uint8);
    # merge nuclear mask, contour, and centroids (detect)
    mask = mask + contour*2 + detect*4;

    Image.fromarray(image).save('./output/image/{}.png'.format(im_no));
    Image.fromarray(refer).save('./output/refer/{}.png'.format(im_no));
    Image.fromarray(mask).save('./output/mask/{}.png'.format(im_no));
    #Image.fromarray(source).save('./output/source/{}.png'.format(im_no));
    #Image.fromarray(detect).save('./output/detect/{}.png'.format(im_no));
    #Image.fromarray(contour).save('./output/contour/{}.png'.format(im_no));
    #Image.fromarray(nucl).save('./output/nucl/{}.png'.format(im_no));
    #Image.fromarray(cyto).save('./output/cyto/{}.png'.format(im_no));
    #Image.fromarray(intp_mask).save('./output/intp_mask/{}.png'.format(im_no));
    fid.write('{} {}\n'.format(im_no, tile_path));
    im_no += im_no_inc;

fid.close();

