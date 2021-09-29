import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from glob import glob
import os
segment_image_directory = '/home/ravi/rsna-miccai-brain-tumor-radiogenomic-classification/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/nnUNet_raw_data/Task873_brats/output_infer/*gz'
Flair_image_directory = '/home/ravi/rsna-miccai-brain-tumor-radiogenomic-classification/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/nnUNet_raw_data/Task873_brats/imagesTr'
Resnet_inputs_directory ='/home/ravi/rsna-miccai-brain-tumor-radiogenomic-classification/resnet_inputs'
for i in glob(segment_image_directory):
  segment_image = i
  #print(segment_image)
  patient_name_flair = i[-12:][:5]+'_0003.nii.gz'
  patient_name_flair = os.path.join(Flair_image_directory,patient_name_flair)
  #print(patient_name_flair)
  segment_image = nib.load(segment_image).get_fdata()
  Flair_image = nib.load(patient_name_flair).get_fdata()
  a = segment_image.shape
  #print(a)
  slices = segment_image.shape[-1]
  count = 0
  k = 0
  for j in range(slices):
    if (np.sum(segment_image[:, :, j])>0):
      count +=1
      
  final_channel_image = np.zeros((a[0],a[1],count,1,2))
  for j in range(slices):
    if (np.sum(segment_image[:, :, j])>0):
      final_channel_image[:,:,k,0,0]=Flair_image[:,:,j]
      final_channel_image[:,:,k,0,1]=segment_image[:, :, j]
      k +=1

  #plt.imshow(final_channel_image[:, :, 40,0,0])

  img = nib.Nifti1Image(final_channel_image, np.eye(4))  # Save axis for data (just identity)
  output_path = os.path.join(Resnet_inputs_directory,i[-12:])
  print(output_path)
  #img.header.get_xyzt_units()
  img.to_filename(output_path)