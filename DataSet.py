import torch.utils.data as Data
import torch
import SimpleITK as sitk
import numpy as np
import os

def resize(ori_image,target_size=(320,160,40),resample_method = sitk.sitkNearestNeighbor):
    '''
    ori_image:original image
    target_size:the size of resampled image
    resample_method:method used when resampling images(default is nearest neighbor)

    return:image after resample
    '''
    
    resampler = sitk.ResampleImageFilter()

    resampler.SetReferenceImage(ori_image)
    # change the voxel size of MRI image to prevent incomplete display
    ori_space = ori_image.GetSpacing()
    ori_size = ori_image.GetSize()
    # print("original image space:",ori_space,"original image size:",ori_size)
    # print("original size:",ori_size,"target size:",target_size)
    target_space = [ori_size[i] / target_size[i] * ori_space[i] for i in range(len(target_size))]
    target_space = tuple(target_space)
    # print("target space:",target_space)

    # set the information of the target image
    resampler.SetSize(target_size)
    resampler.SetOutputOrigin(ori_image.GetOrigin())
    resampler.SetOutputSpacing(target_space)
    resampler.SetOutputDirection(ori_image.GetDirection())
    # print(ori_image.GetSize())
    resampler.SetOutputPixelType(sitk.sitkFloat32)

    # resampler.SetTransform(sitk.Transform(3,sitk.sitkIdentity))
    resampler.SetInterpolator(resample_method)

    resampled_image = resampler.Execute(ori_image)
    return resampled_image


def get_data(path,mcad_info):
    img_names = []
    pathes = []
    labels = []
    img_sites = []
    
    site_dict = {0:"AD_S01",1:"AD_S02",2:"AD_S03",3:"AD_S04",4:"AD_S05",5:"AD_S06",6:"AD_S07"}
    
    sites = os.listdir(path)
    # make sure the number of site is 1 to 7
    sites = sorted(sites)
    # print(sites)
    for site in sites:
        # skip files starting with .
        if site[0] != '.':
            site_img_names = []
            site_path = path + site + '/'
            for img_name in os.listdir(site_path):
                if img_name[0] != '.':
                    site_img_names.append(img_name)
            img_names.append(site_img_names)
    # print(len(img_names))
    
    for i in range(len(mcad_info)):
        img_group = mcad_info.Group[i]
        if img_group == 1 or img_group == 3:
            # 1:normal
            # 3:illness
            # make the number of site start with 0
            img_site = mcad_info.center[i] - 1
            img_name = mcad_info.Subj_t1[i]

            
            if img_name in img_names[img_site]:
                img_path = path + site_dict[img_site] + '/' + img_name + '/image_crop.nii.gz'
                pathes.append(img_path)
                # label = 0 -> normal
                # label = 1 -> illness
                label = 0 if img_group == 1 else 1
                labels.append(label)
                # print(img_site)
                img_sites.append(img_site)
            else:
                print("not found",img_site,img_name)
                
    pathes = np.array(pathes)
    labels = np.array(labels)
    img_sites = np.array(img_sites)

    return pathes,labels,img_sites


class BrainDataSet(Data.Dataset):
    def __init__(self, 
                 img_pathes = None, 
                 labels = None,
                 transform = None, 
                 target_size = None, 
                 sites = None):
        '''
        img_pathes:pathes of all training data(MRI images)
        labels:the label of data
        target_size:unify all the image size into target_size(used in resize function)
        sites:the image belongs to whitch site(used in training process)
        '''
        super().__init__()
        self.img_pathes = img_pathes
        self.labels = labels
        self.transform = transform
        self.target_size = target_size
        self.sites = sites


    def __len__(self):
        return len(self.img_pathes)
    
    def __getitem__(self, index):
        img = sitk.ReadImage(self.img_pathes[index])
        if self.target_size != None:
            # need to resize the image
            img = resize(img, target_size=self.target_size)

        img = sitk.GetArrayFromImage(img)
        label = self.labels[index]
        site = self.sites[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, label, site
