from sys import argv
import os

script, txt_dirs, out_dir = argv
my_list = open(txt_dirs, 'r')
for folder in my_list:
    folder = folder.strip()
    new_folder = out_dir + '/' + folder
    print('Folder created: ' + new_folder)
    if not os.path.exists(new_folder):
        os.makedirs(str(new_folder))
my_list.close()
