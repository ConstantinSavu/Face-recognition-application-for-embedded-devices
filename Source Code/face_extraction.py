import cv2
import os

FILE_PATH = "dataset"

face_extractor = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def read_image(file_name, read_path):
    read_path = os.path.join(read_path, file_name)
    image_data = cv2.imread(read_path)

    return image_data

def save_image(image_data, file_name, save_path):
    save_path = os.path.join(save_path, file_name)
    cv2.imwrite(save_path, image_data)
    



def extract_faces(frame):
    
    gray_image =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces = face_extractor.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    cropped_faces = []

    # detect the face
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        cropped_faces.append(crop_img)
    
    return cropped_faces
    
def extract_all_faces_from_dir(source_dir, dest_dir):
    
    files = (file for file in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, file)))
    
    files = list(files)
    files.sort()
    
    if not os.path.exists(dest_dir): 
        os.mkdir(dest_dir)


    for file in files:
        frame = read_image(file, source_dir)
        cropped_faces = extract_faces(frame)

        for image in cropped_faces:
            save_image(image, file, dest_dir)

def extract_from_all_dirs(source_dir, dest_dir_name):
    dirs = (file for file in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, file)))

    dirs = list(dirs)
    dirs.sort()

    for dir in dirs:
        extract_all_faces_from_dir(
            os.path.join(source_dir, dir),
            os.path.join(source_dir, dir, dest_dir_name)
        )

extract_from_all_dirs('dataset/all_images', "faces")
extract_from_all_dirs('dataset/references', "faces")
extract_from_all_dirs('dataset/test', "faces")