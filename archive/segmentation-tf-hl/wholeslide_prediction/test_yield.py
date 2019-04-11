
image_folder='/nfs/data01/shared/mazhao6/segmentation-tensorflow/pred_on_wholeslide/openslide_svs_extract_4k_tiles/tiles/BC_056_0_1.svs/'
tile_list = image_folder + '/image_resize_list.txt';
lines = [line.strip() for line in open(tile_list, 'r')];
group_size=3
def batch(lines,group_size):
    for index in range(0,11,group_size):
        if index+group_size<10:
          yield range(index,index+group_size)
        else:
          yield range(index,11)

def load_data_group_mao(image_folder,group_size):
  X = [];
  F = [];
  R = [];
  tile_list = image_folder + '/image_resize_list.txt';
  lines = [line.strip() for line in open(tile_list, 'r')];
  for start_index in range(0,len(lines)/group_size,group_size):
    if start_index+group_size<len(lines):
      lines_group=lines[start_index:start_index+group_size]
      #print(start_index,start_index+group_size)
      yield (start_index,start_index+group_size)
    else:
      lines_group=lines[start_index:len(lines)]
      yield (start_index,len(lines))
    
for dd in load_data_group_mao(image_folder,10):
  print(dd)