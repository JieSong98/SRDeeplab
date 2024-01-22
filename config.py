import os

class Config():

    def __init__(self, flag):
        self.n_classes = 2
        self.get_attr(flag)

    def get_attr(self,flag):

        if flag.lower() == 'train':
            self.epoch = 1000
            self.batch_size = 8
            self.learning_rate = 0.001
            self.train_evi_path = r''
            self.train_label_path = r''

            self.val_evi_path = r''
            self.val_label_path = r''

            self.data_augment = True
            self.weight_path = 'weights/aspp'
            self.train_number = self.get_train_number()
            self.val_number = self.get_val_number()
            self.steps_per_epoch = self.train_number//self.batch_size if self.train_number % self.batch_size ==0 else self.train_number//self.batch_size +1
            self.validation_steps= self.val_number//self.batch_size if self.val_number % self.batch_size ==0 else self.val_number// self.batch_size +1

        if flag.lower() == 'test':
            self.batch_size = 16
            self.evi_path = r'/'
            self.weight_path = r'.h5'

            self.output_path= r'/'
            self.image_number = self.get_test_number()
            self.steps = self.image_number//self.batch_size if self.image_number % self.batch_size ==0 else self.image_number//self.batch_size +1

    def check_folder(self,dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def get_train_number(self):
        res = 0
        for dir_entry in os.listdir(self.train_evi_path):
            if os.path.isfile(os.path.join(self.train_evi_path, dir_entry)):
                res += 1
        return res

    def get_val_number(self):
        res = 0
        for dir_entry in os.listdir(self.val_evi_path):
            if os.path.isfile(os.path.join(self.val_evi_path, dir_entry)):
                res += 1
        return res

    def get_test_number(self):
        res = 0
        for dir_entry in os.listdir(self.evi_path):
            if os.path.isfile(os.path.join(self.evi_path, dir_entry)):
                res += 1
        return res






