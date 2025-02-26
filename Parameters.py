import os

class Parameters:
    def __init__(self):
        self.base_dir = './antrenare'
        self.dir_pos_examples_dad = os.path.join(self.base_dir, 'dad')
        self.dir_pos_examples_deedee = os.path.join(self.base_dir, 'deedee')
        self.dir_pos_examples_dexter = os.path.join(self.base_dir, 'dexter')
        self.dir_pos_examples_mom = os.path.join(self.base_dir, 'mom')
        self.dir_pos_examples = [self.dir_pos_examples_dad, self.dir_pos_examples_deedee, 
                                    self.dir_pos_examples_dexter, self.dir_pos_examples_mom]

        self.dir_neg_examples = os.path.join(self.base_dir, 'exempleNegative')
        self.dir_pos_examples_cropped = os.path.join(self.base_dir, 'exemplePozitive')
        # self.dir_test_examples = os.path.join(self.base_dir,'exempleTest/CMU+MIT')  # 'exempleTest/CursVA'   'exempleTest/CMU+MIT'
        self.dir_test_examples = "./validare/validare"

        # self.path_annotations = os.path.join(self.base_dir, 'exempleTest/CMU+MIT_adnotari/ground_truth_bboxes.txt')
        self.path_annotations_dad = os.path.join(self.base_dir, 'dad_annotations.txt')
        self.path_annotations_deedee = os.path.join(self.base_dir, 'deedee_annotations.txt')
        self.path_annotations_dexter = os.path.join(self.base_dir, 'dexter_annotations.txt')
        self.path_annotations_mom = os.path.join(self.base_dir, 'mom_annotations.txt')
        self.path_annotations = [self.path_annotations_dad, self.path_annotations_deedee, 
                                    self.path_annotations_dexter, self.path_annotations_mom]

        self.path_annotations_test = "./validare/validare_annotations.txt"
        # self.dir_save_files = os.path.join(self.base_dir, 'salveazaFisiere')
        self.dir_save_files = "./evaluare/fisiere_solutie/461_Andronic_Smaranda/task1"
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        # set the parameters
        self.dim_window = 36  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
        self.dim_hog_cell = 6  # dimensiunea celulei
        self.dim_descriptor_cell = 36  # dimensiunea descriptorului unei celule
        self.overlap = 0.3
        self.number_positive_examples = 5813  # numarul exemplelor pozitive
        self.number_negative_examples = 11988  # numarul exemplelor negative
        self.overlap = 0.3
        self.has_annotations = True
        self.threshold = 0
