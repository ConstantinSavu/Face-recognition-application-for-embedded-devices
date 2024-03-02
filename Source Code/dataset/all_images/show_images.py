import cv2
import matplotlib.pyplot as plt
import os

def read_image(file_name, read_path):
    read_path = os.path.join(read_path, file_name)
    image_data = cv2.imread(read_path)

    return image_data


def show_from_all_dirs(source_dir, additional):
    dirs = (file for file in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, file)))

    dirs = list(dirs)
    dirs.sort()

    fig, axs = plt.subplots(nrows=5, ncols=9, figsize=(30, 30))

    cols = ["Reference image", "Reference image", "Reference image", "Reference image", "Reference image", "Reference image", "Reference image", 
            "Test image", "Test image"]
    
    rows = ["Diego Luna", "Gael Garcia Bernal", "Guillermo del Toro", "Selma Hayek", "Sergio Perez"]

    pad = 5
    for ax, col in zip(axs[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    for ax, row in zip(axs[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

    for i, dir in enumerate(dirs):
        
        files = (file for file in os.listdir(os.path.join(source_dir, dir, additional)) if os.path.isfile(os.path.join(source_dir, dir, file)))
        
        files = list(files)
        files.sort()
        

        for j, file in enumerate(files):
            frame = read_image(file, os.path.join(source_dir, dir, additional))
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            axs[i, j].imshow(frame)
            axs[i, j].tick_params(left=False, labelleft=False)
            axs[i, j].xaxis.set_visible(False)

    

show_from_all_dirs(".", "")
show_from_all_dirs(".", "faces")
plt.show()