import numpy as np 
import cv2
import pandas as pd 
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True #https://www.programmersought.com/article/44876269499/
from data_augmentation import *
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
IMAGENET_MEAN = [0.485, 0.456, 0.406]  
IMAGENET_STD = [0.229, 0.224, 0.225]

class CustomDataset(Dataset):
    def __init__(self, folder_dir, dataframe, transformer=None, label_name = None):
        '''
            Args:
            folder_dir: (string) path to data (images) folder
            dataframe: (pandas.dataframe) contatin image's id and annotations
            transform: (object) transformer for FundusDataset
            label_name: (iterable) contain name of diseases in dataset
        '''
        self.folder_dir = folder_dir
        self.df = dataframe
        self.label_name = label_name

        # contatin unique image's id
        self.img_ids = self.df['img_id'].unique()

        self.transformer = transformer
        self.bboxes = self.df[['x_min', 'y_min', 'x_max', 'y_max']].values
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_labels = self.df[self.df['img_id'] == img_id]['labels'].values
        img_bboxes = self.df[self.df['img_id'] == img_id][['x_min', 'y_min', 'x_max', 'y_max']].values
        img = cv2.imread(os.path.join(self.folder_dir, img_id))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transformer:
            img, img_bboxes, _ = self.transformer(img, img_bboxes, img_labels)
    
        return img, img_bboxes

class CustomTransformer(object):
    def __init__(self, augmentations, probs=1):
        self.augmentations = augmentations
        self.probs = probs
    
    def __call__(self, img, bboxes, labels):
        augmentations = Sequence_(augmentations = self.augmentations, probs = self.probs)
        compose = transforms.Compose([transforms.ToTensor(), 
                                      transforms.Normalize(mean = IMAGENET_MEAN, std = IMAGENET_STD)])
        # augmentations
        img, bboxes, labels = augmentations(img, bboxes, labels)

        # compose image
        img = compose(img)
        
        # convert it to float32 (for backward)
        bboxes = bboxes.astype(np.float32)
        #labels = labels.astype(np.float32)

        # convert bboxes and labels to tensor
        bboxes = torch.from_numpy(bboxes)
        #labels = torch.from_numpy(labels)

        return img, bboxes, labels

if __name__ == "__main__":
    folder_dir = "./dataset"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    annotation_file = os.path.join("./dataset", "annotations.csv")
    transformer = CustomTransformer(augmentations = [RandomHorizontalFlip_(p=1), Rotate(angle = 45, p=1), 
                                                     Shear(-0.2, p=1), Resize((256, 256))], 
                                    probs = 1)
    df = pd.read_csv(annotation_file)
    dataset = CustomDataset(folder_dir = folder_dir, dataframe = df, 
                            transformer = transformer, label_name = ['Cardiomegaly', 'Aortic enlargement', 'Pleural thickening'])
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = False)
    '''
    img, bboxes = dataset[0]
    print(img.shape)
    print(bboxes)
    print(labels)
    '''
    for img, bboxes in dataloader:
        img, bboxes = img.to(device), bboxes.to(device)
        print(img.shape)
        print(bboxes)
        

