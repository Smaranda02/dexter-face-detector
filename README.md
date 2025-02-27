# **Dexter's Laboratory Face Detector**  
A face detection project using a **multiscale sliding window** approach.

---

## **📦 Required Libraries**  
Ensure you have the following dependencies installed:  

```plaintext
numpy==2.2.1  
opencv-python==4.10.0.84  
scikit-image==0.25.0  
scikit-learn==1.6.0  
imageio==2.36.1  
ultralytics==8.3.63  
```

You can install them with:  
```bash
pip install -r requirements.txt
```

---

## **🚀 Running the Project**  
### **🔹 Standard Detection Approach**
To run the face detection model:  
```bash
python RunProject.py <validation_folder_name>
```
- `<validation_folder_name>` is the path to the folder containing test images.  
- If not specified, the default folder is `"validare/validare"`  

#### **Example usage:**  
```bash
python RunProject.py validare/validare  
python RunProject.py evaluare/fake_test  
```

### **🔹 Running YOLO for Face Detection**
To use the **YOLO model** for face detection:  
```bash
python YOLO.py <validation_folder_name>
```

---

## **📂 Output Files**  
- **All output files** are stored in:  
  `evaluare/fisiere_solutie/461_Andronic_Smaranda/task1/`
- **For standard detection**, the folder contains:  
  - Images with **bounding boxes** for detected faces  
  - An **annotation file**  
- **For YOLO detections**, files are saved in the same directory, with the prefix:  
  - **`YOLO_file_names_all_faces.npy`**  

---

## **🖼️ Detection Examples**  
Here are some sample detection results:

![Detection Example 1](https://github.com/user-attachments/assets/248ee38f-1004-4755-81db-29fb8fb3037b)  

![Detection Example 2](https://github.com/user-attachments/assets/816f42e6-1d48-4a1c-a547-2ea00722023a)  

---

