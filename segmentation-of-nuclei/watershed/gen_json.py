import json
import os
import collections
import sys
import datetime

dict_version = {};
dict_version['git_remote'] = os.environ.get('GIT_REMOTE')
dict_version['git_branch'] = os.environ.get('GIT_BRANCH')
dict_version['git_commit'] = os.environ.get('GIT_COMMIT')
dict_version['model_path'] = os.environ.get('MODEL_PATH')
dict_version['model_hash'] = os.environ.get('MODEL_HASH')
dict_version['model_url']  = os.environ.get('MODEL_URL')
dict_version['model_ver']  = os.environ.get('MODEL_VER')

analysis_id = 'CNN_synthetic_n_real_'+os.environ.get('SEG_VERSION')
analysis_execution_date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

def gen_meta_json(in_path, image_id, wsi_width, wsi_height, method_description,
        seg_thres, det_thres, win_size, min_nucleus_size, max_nucleus_size):
    file_id = os.path.basename(in_path)[: -len('_SEG.png')]
    fields = file_id.split('_')
    x = int(fields[0])
    y = int(fields[1])
    size1 = int(fields[2])
    size2 = int(fields[3])
    mpp = float(fields[4])

    dict_model = collections.OrderedDict()
    dict_model['input_type'] = 'wsi'
    dict_model['otsu_ratio'] = 0.0
    dict_model['curvature_weight'] = 0.0
    dict_model['min_size'] = min_nucleus_size
    dict_model['max_size'] = max_nucleus_size
    dict_model['ms_kernel'] = 0
    dict_model['declump_type'] = 0
    dict_model['levelset_num_iters'] = 0
    dict_model['mpp'] = mpp
    dict_model['image_width'] = wsi_width
    dict_model['image_height'] = wsi_height
    dict_model['tile_minx'] = x
    dict_model['tile_miny'] = y
    dict_model['tile_width'] = size1
    dict_model['tile_height'] = size2
    dict_model['patch_minx'] = x
    dict_model['patch_miny'] = y
    dict_model['patch_width'] = size1
    dict_model['patch_height'] = size2
    dict_model['output_level'] = 'mask'
    dict_model['out_file_prefix'] = file_id
    dict_model['subject_id'] = image_id
    dict_model['case_id'] = image_id
    dict_model['analysis_id'] = analysis_id 
    dict_model['analysis_execution_date'] = analysis_execution_date
    dict_model['version'] = dict_version
    dict_model['analysis_desc'] = '{}_{}_{}_{}_{}_{}'.format(
            method_description, seg_thres, det_thres, win_size, min_nucleus_size, max_nucleus_size)

    json_str = json.dumps(dict_model)

    fid = open(os.path.join(os.path.dirname(in_path), file_id+'-algmeta.json'), 'w')
    fid.write(json_str)
    fid.close()

