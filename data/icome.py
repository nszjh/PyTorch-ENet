import os
from collections import OrderedDict
import torch.utils.data as data
from . import utils


class Icome(data.Dataset):
    """CamVid dataset loader where the dataset is arranged as in

    Keyword arguments:
    - root_dir (``string``): Root directory path.
    - mode (``string``): The type of dataset: 'train' for training set, 'val'
    for validation set, and 'test' for test set.
    - transform (``callable``, optional): A function/transform that  takes in
    an PIL image and returns a transformed version. Default: None.
    - label_transform (``callable``, optional): A function/transform that takes
    in the target and transforms it. Default: None.
    - loader (``callable``, optional): A function to load an image given its
    path. By default ``default_loader`` is used.

    """
    # Training dataset root folders
    train_folder = 'icome/icome_task2_data/clean_images/images'
    train_lbl_folder = 'icome/icome_task2_data/clean_images/profiles/pre_change'

    # Validation dataset root folders
    val_folder = "icome/icome_test_images/test_images"
    val_lbl_folder = "icome/icome_test_images/testornot_images/pre_change"

    # Test dataset root folders
    test_folder = 'icome/icome_test_images/test_images'
    test_lbl_folder = 'icome/icome_test_images/testornot_images/pre_change'

    renamedTest_folder = 'icome/icome_test_images/s06_e02'

    # Images extension
    img_extension = '.jpg'
    img_label_extension = '.png'

    # Default encoding for pixel value, class name, and class color
    color_encoding = OrderedDict([
        ('person', (128, 128, 128)),
        ('unlabeled', (255, 255, 255))
    ])

    def __init__(self,
                 root_dir,
                 mode='train',
                 transform=None,
                 label_transform=None,
                 loader=utils.pil_loader):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.label_transform = label_transform
        self.loader = loader
        
        print (self.mode)

        if self.mode.lower() == 'train':
            # Get the training data and labels filepaths
            self.train_data = utils.get_files(
                os.path.join(root_dir, self.train_folder),
                extension_filter=self.img_extension)

            self.train_labels = utils.get_files(
                os.path.join(root_dir, self.train_lbl_folder),
                extension_filter=self.img_label_extension)
        
        elif self.mode.lower() == 'val':
            # Get the validation data and labels filepaths
            self.val_data = utils.get_files(
                os.path.join(root_dir, self.val_folder),
                extension_filter=self.img_extension)

            self.val_labels = utils.get_files(
                os.path.join(root_dir, self.val_lbl_folder),
                extension_filter=self.img_label_extension)


        elif self.mode.lower() == 'test':
            # Get the test data and labels filepaths
            self.test_data = utils.get_files(
                os.path.join(root_dir, self.test_folder),
                extension_filter=self.img_extension)

            self.test_labels = utils.get_files(
                os.path.join(root_dir, self.test_lbl_folder),
                extension_filter=self.img_label_extension)

        elif self.mode.lower() == 'renamedtest':
            # Get the test data and labels filepaths
            self.renamedTest_data = utils.get_files(
                os.path.join(root_dir, self.renamedTest_folder),
                extension_filter=self.img_extension)

        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

    def __getitem__(self, index):
        """
        Args:
        - index (``int``): index of the item in the dataset

        Returns:
        A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
        of the image.

        """
        isRenamed = 0
        if self.mode.lower() == 'train':
            data_path, label_path = self.train_data[index], self.train_labels[
                index]
        elif self.mode.lower() == 'val':
            data_path, label_path = self.val_data[index], self.val_labels[
                index]
        elif self.mode.lower() == 'test':
            data_path, label_path = self.test_data[index], self.test_labels[
                index]
        elif self.mode.lower() == 'renamedtest':
            data_path = self.renamedTest_data[index]
            isRenamed = 1
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

        if isRenamed == 1:
            img = utils.pil_loader2(data_path)
            label = ''
            if self.transform is not None:
                img = self.transform(img)
        else:
            # print (data_path, label_path)
            img, label = self.loader(data_path, label_path)

            # # print (label.size)
            # # print ("-------\n")
            # label1 = label.convert('1')
            # label1 = label1.convert('L')
            
            # # print (label.size)        
            # label1 = utils.fliterLabel(label)
            # filePath, fileName = os.path.split(label_path)
            # label1.save(filePath + "/pre_change/" + fileName.split('.')[0] + '.png')
            
            if self.transform is not None:
                img = self.transform(img)

            if self.label_transform is not None:
                label = self.label_transform(label)
        

        return img, label

    def __len__(self):
        """Returns the length of the dataset."""
        if self.mode.lower() == 'train':
            return len(self.train_data)
        elif self.mode.lower() == 'val':
            return len(self.val_data)
        elif self.mode.lower() == 'test':
            return len(self.test_data)
        elif self.mode.lower() == 'renamedtest':
            return len(self.renamedTest_data)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")
