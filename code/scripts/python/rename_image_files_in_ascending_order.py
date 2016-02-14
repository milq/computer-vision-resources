import glob
import os

id_frame = 0
folder = 'path_folder'

list_files = sorted(glob.glob(folder + '/*.png'))

for i in list_files:   
    if id_frame % 2 == 0:
        new_file = folder + '/' + str(id_frame).zfill(5) + 'a.png'
    else:
        new_file = folder + '/' + str(id_frame).zfill(5) + 'b.png'        
    id_frame = id_frame + 1
    os.rename(i, new_file)
