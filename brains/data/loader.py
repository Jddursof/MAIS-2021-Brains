#no data allowed to be uploaded :,(

from torch.utils.data import Dataset
import os
from glob import glob
import nibabel as nib
import numpy as np
import pandas as pd

MODALS = {
    'FLR':'_flair.nii.gz',
    'SEG':'_seg.nii.gz',
    'T1C':'_t1ce.nii.gz',
    'T1':'_t1.nii.gz',
    'T2':'_t2.nii.gz'
}

# for testing purposes
ROOT_DIR = '/usr/local/faststorage/BraTS19_Data/'

def build_dataset(root_dir,dataset_type,mode=MODALS["T1"]):
    if (dataset_type == 'train'):
        root_dir = f"{root_dir}Training"    
        train_val = 'Training'
        clinical_dir = f"{root_dir}/survival_data.csv"
    elif (dataset_type == 'val'):
    #else:
        root_dir = f"{root_dir}Validation"
        train_val = 'Validation'
        clinical_dir = f"{root_dir}/survival_evaluation.csv"
    data_dir = f"{root_dir}/Data"
    mode = mode
    clinical_df = pd.read_csv(clinical_dir)
    usr_dirs = glob(f"{data_dir}/*")
    output = {}
    dataset=[]
    for usr in usr_dirs:
        usr_key = os.path.basename(os.path.normpath(usr))
        if(clinical_df['BraTS19ID']==str(usr_key)).any():
            usr_clinical = clinical_df.loc[clinical_df['BraTS19ID'] == usr_key]
            if not np.isnan(float(usr_clinical['Age'].item())):
                dataset.append(usr)
    return dataset
            

class VolumeLoader(Dataset):
    def __init__(self, dataset, root_dir, dataset_type, mode=MODALS['T1']):
        if (dataset_type == 'train'):
            self.root_dir = f"{root_dir}Training"    
            self.train_val = 'Training'
            self.clinical_dir = f"{self.root_dir}/survival_data.csv"
        elif (dataset_type == 'val'):
            self.root_dir = f"{root_dir}Validation"
            self.train_val = 'Validation'
            self.clinical_dir = f"{self.root_dir}/survival_evaluation.csv"     
        
        
        self.data_dir = f"{self.root_dir}/Data"
        self.mode = mode
        self.clinical_df = pd.read_csv(self.clinical_dir)
        self.dataset=dataset
        
            

    def __len__(self):
        '''
            get num of imaging files/dirs
        '''
        return len(self.dataset)    
    def __getitem__(self, idx):
        # get mri dir, import as numpy arr, add to dict
        usr=self.dataset[idx]
        usr_key = os.path.basename(os.path.normpath(usr))
        usr_mri_dir = f"{usr}/{usr_key}{self.mode}"
        usr_dict={}
        usr_dict['MRI'] = nib.load(usr_mri_dir).get_fdata()
        
        # get clinical for user and add to dict
        usr_clinical = self.clinical_df.loc[self.clinical_df['BraTS19ID'] == usr_key]
        usr_dict['Age'] = usr_clinical['Age'].item()
        #usr_dict['ResectionStatus'] = usr_clinical['ResectionStatus']
        #if (self.train_val == 'Training'):      # only present in training data
        #    usr_dict['Survival'] = usr_clinical['Survival']
        return usr_dict
        
