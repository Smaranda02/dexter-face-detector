from ultralytics import YOLO
import os
import numpy as np
import sys

if len(sys.argv) < 2:
    print("Usage: python RunProject.py <file_name>")
    sys.exit(1)

test_images_folder = sys.argv[1]

model = YOLO('./runs/detect/yolov8n_custom/weights/best.pt') 
output_folder = "./evaluare/fisiere_solutie/461_Andronic_Smaranda/task1"

os.makedirs(output_folder, exist_ok=True) 

test_file_names = os.listdir(test_images_folder)
test_file_names.sort(key=lambda x: int(os.path.splitext(x.replace("image_", ""))[0])) #we sort the files
file_paths = [os.path.join(test_images_folder, f) for f in test_file_names] 

detections = []  # array cu toate detectiile pe care le obtinem
scores = []  # array cu toate scorurile pe care le obtinem
file_names = []

results = model.predict(source=file_paths, save=False)  #results contine prezicerile facute


for result in results:

    for box in result.boxes:
        bbox = list(map(int, box.xyxy[0].tolist()))
        detections.append(bbox)

        old_name = os.path.basename(result.path)
        base_name = os.path.splitext(old_name)[0]  # Remove extension
        numeric_part = base_name.split('_')[-1]   # Extract the number after the underscore
        new_name = f"{int(numeric_part):03d}.jpg" #we fill with zeros like 001.jpg instead of 1.jpg
        file_names.append(new_name)
        
        confidence = box.conf.item()
        scores.append(confidence)

np.save(os.path.join(output_folder, "YOLO_file_names_all_faces.npy"), np.array(file_names))  
np.save(os.path.join(output_folder, "YOLO_detections_all_faces.npy"), np.array(detections)) 
np.save(os.path.join(output_folder, "YOLO_scores_all_faces.npy"), np.array(scores)) 

# print(file_names, detections, scores)

