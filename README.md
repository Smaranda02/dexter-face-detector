## Dexter's Laboratory face detector using multiscale sliding window

### 1. The libraries required to run the project including the full version of each library.

- numpy==2.2.1
- opencv_python==4.10.0.84
- scikit_image==0.25.0
- scikit_learn== 1.6.0
- imageio  == 2.36.1
- ultralytics === 8.3.63

### 2. How to run your code and where to look for the output files.
  
- how to run : python RunProject.py <validation_folder_name>, where validation_folder_name is the path to the folder containing the test images
             - the default is "validare/validare
             - python RunProject.py validare/validare or python RunProject.py evaluare/fake_test
- output: the output files for all the tasks are found in folder evaluare/fisiere_solutie/461_Andronic_Smaranda/task1,
        where images with bounding boxes for detections and the annotation file are found 

### For running the YOLO model for task1, the same approach applies:
- how to run : python YOLO.py <validation_folder_name> 
- output: the output files are found in folder evaluare/fisiere_solutie/461_Andronic_Smaranda/task1, and are preceded 
        by the prefix "YOLO" : “YOLO_file_names_all_faces.npy”

