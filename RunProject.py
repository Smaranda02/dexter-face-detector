from Parameters import *
from FacialDetector import *
from Visualize import *
import random
import sys
import shutil

params: Parameters = Parameters()
params.dim_window = 64
params.dim_hog_cell = 8  # dimensiunea celulei
params.overlap = 0.3
params.number_positive_examples = 5813  # numarul exemplelor pozitive
params.number_negative_examples = 11707  # numarul exemplelor negative

params.threshold = 2.5 # toate ferestrele cu scorul > threshold si maxime locale devin detectii
params.has_annotations = True

params.use_hard_mining = False  # (optional)antrenare cu exemple puternic negative
params.use_flip_images = True  # adauga imaginile cu fete oglindite

if params.use_flip_images:
    params.number_positive_examples *= 2

facial_detector: FacialDetector = FacialDetector(params)

# Pasii 1+2+3. Incarcam exemplele pozitive (cropate) si exemple negative generate
# verificam daca sunt deja existente

def createYOLODataset():
     
     trainFolder = ".\\dataset\\images\\train"
     currentIndex = 0

     for index in range(4):
        currentIndex = 1000 * index 
        positive_folder =  params.dir_pos_examples[index]
        images_path = os.path.join(positive_folder, '*.jpg')
        files = glob.glob(images_path)

        for file in files:
            currentIndex += 1
            positive_image = cv.imread(file, cv.IMREAD_COLOR)
            file_name = f"image_{currentIndex}.jpg"
            file_path = os.path.join(trainFolder, file_name)
            success = cv.imwrite(file_path, positive_image)
            if not success:
                print(f"Error: Could not write image to {file_path}")

# createYOLODataset()

def createYOLOLabels():
    # labelFolder =  ".\\dataset\\labels\\train"
    labelFolder =  ".\\dataset\\labels\\val"

    positive_index = 1
    if not os.path.exists(labelFolder):
        os.makedirs(labelFolder) 

    for index in range(1):
        image_number = 1
        # annotations_file = params.path_annotations[index]
        annotations_file = '.\\validare\\validare_annotations.txt'
        face_positions = []
        with open(annotations_file, "r") as annotations:
            for line in annotations:
                stripped_line = line.strip()
                # Split the line by spaces
                parsed_line = stripped_line.split()
                image_index = parsed_line[0].split(".jpg")[0] #we get 0001, 0002, ..., 0100, ...
                nr_digits = len(str(image_number))
                w, h, x, y = map(int, parsed_line[1:5])  # Convert to integers
                x_center = round((x + w) / 960, 2)
                y_center = round((y + h) / 720, 2)
                width = round((x - w) / 480, 2)
                height = round((y - h) / 360, 2)

                if image_index[-nr_digits:] == str(image_number):  #we are still on the adnotatins of the same image
                    #we are still on the same image which has multiple annotations 
                    # positions = [w,h,x,y]
                    positions = [x_center, y_center, width, height]
                    face_positions.append(positions)
                
                else:
                    nr = image_number + index * 1000
                    file_name = f"image_{nr}.txt"
                    file_path = os.path.join(labelFolder, file_name)
                    with open(file_path, "w") as label_file:
                        for pos in face_positions:
                            print(nr, pos)
                            label_file.write("0 " + " ".join(map(str, pos)) + "\n")

                    image_number += 1
                    positive_index += 1 #for all the 4000 images
                    face_positions = []
                    # positions = [w,h,x,y]
                    positions = [x_center, y_center, width, height]
                    face_positions.append(positions)

            nr = image_number + index * 1000
            file_name = f"image_{nr}.txt"
            file_path = os.path.join(labelFolder, file_name)
            with open(file_path, "w") as label_file:
                for pos in face_positions:
                    print(nr, pos)
                    label_file.write("0 " + " ".join(map(str, pos)) + "\n")

# createYOLOLabels()

def createCroppedFacesImages():
    negative_index = 0
    for index in range(4):
        positive_folder =  params.dir_pos_examples[index]
        annotations_file = params.path_annotations[index]
        images_path = os.path.join(positive_folder, '*.jpg')
        files = glob.glob(images_path)
        image_number = 1
        while True:
            line_index = 1
            face_positions = []
            with open(annotations_file, "r") as adnotations:
                for line in adnotations:
                    stripped_line = line.strip()
                    # Split the line by spaces
                    parsed_line = stripped_line.split()
                    image_index = parsed_line[0].split(".jpg")[0] #we get 0001, 0002, ..., 0100, ...
                    nr_digits = len(str(image_number))
                    w, h, x, y = map(int, parsed_line[1:5])  # Convert to integers

                    if image_index[-nr_digits:] == str(image_number):
                        #we are still on the same image which has multiple annotations 
                        positions = [w,h,x,y]
                        face_positions.append(positions)

                    else:
                        negative_patches = 0 
                        ok = 1
                        image = cv.imread(files[image_number - 1], cv.IMREAD_GRAYSCALE)
                        #if out photo isn't big enough to extract 36x36 patches for nagative images we continue
                        if image.shape[0] >= params.dim_window and image.shape[1] >= params.dim_window:
                            while negative_patches < 3:  #for each photo we want to generate 3 negative images 
                                
                                #create random patch
                                height, width = image.shape
                                min_dim = min(width, height)
                                fraction = 4
                                x1 = random.randint(0, width - min_dim // fraction)
                                y1 = random.randint(0, height - min_dim // fraction)
                                random_patch = [x1, y1, x1 + min_dim // fraction, y1 + min_dim // fraction]

                                #iterate over each face in the current image 
                                ok = 1
                                for face_pos in face_positions:
                                    iou = facial_detector.intersection_over_union(random_patch, face_pos)
                                    if iou > 0.015:  #check if our random patch is intersecting too much with a face 
                                        ok = 0
                                        break
                                if ok: #if it didn't intersect with a previous face we store the patch as a negative example
                                    patch = image[y1:y1 + min_dim //fraction , x1:x1 + min_dim // fraction]
                                    patch_filename = os.path.join(params.dir_neg_examples, f"negative_{negative_index}.jpg")

                                    cv.imwrite(patch_filename, patch)
                                    negative_patches += 1
                                    negative_index +=1
                                    print("Negative image : ", negative_index)

                        image_number += 1
                        face_positions = []
                    
                    image = files[image_number - 1]
                    image = cv.imread(image, cv.IMREAD_GRAYSCALE)

                    cropped_image = image[h:y, w:x]
                    file_name = f"cropped_{index + 1}_{image_index}_{line_index}.jpg"
                    file_path = os.path.join(params.dir_pos_examples_cropped, file_name)
                    cv.imwrite(file_path, cropped_image)
                    line_index += 1
            break

# createCroppedFacesImages()      

if len(sys.argv) < 2:
    print("Usage: python RunProject.py <file_name>")
    sys.exit(1)

validation_directory = sys.argv[1]
params.dir_test_examples = validation_directory


positive_features_path = os.path.join(params.dir_save_files, 'descriptoriExemplePozitive_' + str(params.dim_hog_cell) + '_' +
                        str(params.number_positive_examples) + '.npy')

if os.path.exists(positive_features_path):
    positive_features = np.load(positive_features_path)
    print(positive_features.shape)
    print('Am incarcat descriptorii pentru exemplele pozitive')
else:
    print('Construim descriptorii pentru exemplele pozitive:')
    positive_features = facial_detector.get_positive_descriptors()
    np.save(positive_features_path, positive_features)
    print(positive_features.shape)
    print('Am salvat descriptorii pentru exemplele pozitive in fisierul %s' % positive_features_path)


negative_features_path = os.path.join(params.dir_save_files, 'descriptoriExempleNegative_' + str(params.dim_hog_cell) + '_' +
                        str(params.number_negative_examples) + '.npy')

if os.path.exists(negative_features_path):
    negative_features = np.load(negative_features_path)
    print('Am incarcat descriptorii pentru exemplele negative')
    print(negative_features.shape)
else:
    print('Construim descriptorii pentru exemplele negative:')
    negative_features = facial_detector.get_negative_descriptors()
    np.save(negative_features_path, negative_features)
    print(negative_features.shape)
    print('Am salvat descriptorii pentru exemplele negative in fisierul %s' % negative_features_path)

# Pasul 4. Invatam clasificatorul liniar
training_examples = np.concatenate((np.squeeze(positive_features), np.squeeze(negative_features)), axis=0)
train_labels = np.concatenate((np.ones(positive_features.shape[0]), np.zeros(negative_features.shape[0])))
facial_detector.train_classifier(training_examples, train_labels)


detections, scores, file_names = facial_detector.run()
output_folder = params.dir_save_files

output_files = {
    "detections_all_faces.npy": detections,
    "scores_all_faces.npy": scores,
    "file_names_all_faces.npy": file_names
}

for file_name, values in output_files.items():
    file_path = os.path.join(output_folder, file_name)
    print(values)
    np.save(file_path, values)


if params.has_annotations:
    facial_detector.eval_detections(detections, scores, file_names)
    show_detections_with_ground_truth(detections, scores, file_names, params)
else:   
    show_detections_without_ground_truth(detections, scores, file_names, params)

