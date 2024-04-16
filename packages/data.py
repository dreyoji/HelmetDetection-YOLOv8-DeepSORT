# Gathers the dataset with the test set and the training set
import glob
from PIL import Image
from tqdm import tqdm

class dataset:
    def __init__(self, train_path, test_path, valid_path):
        # Create an empty dataset
        self.train_img = []
        self.test_img = []
        self.valid_img = []
        # Get all images within the path
        self.train_images_path = glob.glob(f"{train_path}/images/*.jpg")
        self.test_images_path = glob.glob(f"{test_path}/images/*.jpg")
        self.valid_images_path = glob.glob(f"{valid_path}/images/*.jpg")

        self.train_img = self.get_data(self.train_img, self.train_images_path)
        self.test_img = self.get_data(self.test_img, self.test_images_path)
        self.valid_img = self.get_data(self.valid_img, self.valid_images_path)

    @staticmethod
    def get_data(dataset, path):
        # Load image and put it in a dictionary based on time
        """Gets the images inside a path and returns a dictionary with images inside based on
        the time of day"""
        for file in tqdm(path, desc="Loading images", unit="image"):
            img = Image.open(file)
            dataset.append(img)
        return dataset

