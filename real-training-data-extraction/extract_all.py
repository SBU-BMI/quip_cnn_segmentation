from extract_patches_a_dataset_and import extract_a_dataset
from extract_patches_miccai15 import extract_miccai15
from extract_patches_miccai16 import extract_miccai16

patch_id = 1

print 'extracting a dataset starting at id', patch_id
patch_id = extract_a_dataset(patch_id, stride=31, win_size=400)

print 'extracting miccai15 starting at id', patch_id
patch_id = extract_miccai15(patch_id, stride=31, win_size=400)

print 'extracting miccai16 starting at id', patch_id
patch_id = extract_miccai16(patch_id, stride=31, win_size=400)
print 'all patches extracted', patch_id

