import sys
slide_name =  sys.argv[1];
slide_file_name=sys.argv[1].split('/')
slide_file_name=slide_file_name[len(slide_file_name)-1]
output_folder = sys.argv[2] + '/' + slide_file_name;

print(output_folder)