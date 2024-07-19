import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.misc
import numpy as np
from glob import glob
from skimage.io import imread
import nibabel as nib
from skimage.util import montage 
import argparse

from PIL import Image
import matplotlib
    

parser = argparse.ArgumentParser(description="Swin UNETR segmentation viewer")
parser.add_argument("--dir", default='/home/gpuadmin/yujin/ro-llama/work_dir/PC_NC/v11.0_HU_gtv_1.00', type=str, help="pretrained checkpoint directory") #v1_HN_CTV
parser.add_argument("--test_mode", default=1, type=int)
parser.add_argument("--select_list", default=["943756", "9288308"], type=list) #'HN_P028_08-06-2015-NA-RTHEADNECK Adult-85396_data'
parser.add_argument("--rogpt", default=False)
parser.add_argument("--ablation", default="", type=str)

def main():
    
    args = parser.parse_args()
    
    # args.dir = args.dir.replace('runs', 'outputs')
    
    # tag = ('_rogpt' if args.rogpt else '') + args.ablation + ('_ext1' if args.test_mode == 2 else '')

    folder_list = glob(args.dir + '/raw_SC*')
    print(args.ablation)
    
    for folder in folder_list:
        tag = ('_%s'%folder.split('/')[-1].split('_')[-2])
        all_images = glob(os.path.join(folder, '*_data.nii'))
        if len(args.select_list) > 0:
            all_images = [images for images in all_images if images.split('/')[-1].split('_data.nii')[0] in args.select_list]
            if not os.path.exists(args.dir + '/s_vis' + tag):
                os.makedirs(args.dir + '/s_vis' + tag)
        else:
            if not os.path.exists(args.dir + '/vis' + tag):
                os.makedirs(args.dir + '/vis' + tag)

        for file_name in all_images:
            
            file_index = file_name.split('/')[-1]
            
            # load
            all_masks = glob(os.path.join(file_name.replace('_data.nii','_label.nii')))[0]
            all_preds = glob(os.path.join(file_name.replace('_data.nii','.nii')))[0]
            pred = nib.load(all_preds).get_fdata()
            mask = nib.load(all_masks).get_fdata()
            image = nib.load(file_name).get_fdata()
            
            # lr flip
            image = image[:, -1::-1, ...]
            pred = pred[:, -1::-1, ...]
            mask = mask[:, -1::-1, ...]
            
            # Axial
            mask_indices = np.sum(mask, axis=(0,1))
            mask_indices = np.where(mask_indices>0)[0]
            center = (mask_indices.max() - mask_indices.min())//2 + mask_indices.min() 
            print('[Axial] index range = ', mask_indices.max() - mask_indices.min())

            # X = np.array([[0., 0., 0.],  
            #       [0., 1., 0.],  
            #       [1., 0., 0.]]) #red
            
            image_orig = image[...,center-40:center+40:2] * 0.5
            image_orig = np.expand_dims(image_orig, axis=3).repeat(3,3)
            image_mask = image_orig.copy()
            image_mask[...,0] += mask[...,center-40:center+40:2] * 128
            image_pred = image_orig.copy()
            image_pred[...,0] += pred[...,center-40:center+40:2] * 128
            
            image_orig = np.transpose(image_orig, (2,0,1,3))
            image_mask = np.transpose(image_mask, (2,0,1,3))
            image_pred = np.transpose(image_pred, (2,0,1,3))

            if args.ablation == 'ci':
                image_ci_mask = image_orig.copy()
                cmap = matplotlib.cm.get_cmap('viridis')
                all_cis = glob(os.path.join(file_name.replace('_data.nii','_ci.nii')))[0]
                ci = nib.load(all_cis).get_fdata()
                ci = ci[:, -1::-1, ...]
                image_ci = ci[...,center-40:center+40:2]
                image_ci = (image_ci / 255.0) 
                image_ci = cmap(image_ci) 
                image_ci = np.transpose(image_ci, (2,0,1,3))
                binary = pred[...,center-40:center+40:2]
                binary = np.expand_dims(binary, axis=3).repeat(3,3)
                binary = np.transpose(binary, (2,0,1,3))
                image_ci_mask = image_ci_mask + image_ci[..., :3] * 128 * binary
                # image_ci_mask[image_ci_mask>=254] = 254


            # Coronal
            mask_indices = np.sum(mask, axis=(1,2))
            mask_indices = np.where(mask_indices>0)[0]
            center_c = (mask_indices.max() - mask_indices.min())//2 + mask_indices.min() 
            print('[Coronal] index range = ', mask_indices.max() - mask_indices.min())
            
            image_orig_c = image[center_c-50:center_c+50:2, ...] * 0.5
            image_orig_c = np.expand_dims(image_orig_c, axis=3).repeat(3,3)
            image_pred_c = image_orig_c.copy()
            image_mask_c = image_pred_c.copy()
            image_pred_c[...,0] += pred[center_c-50:center_c+50:2, ...] * 128
            image_mask_c[...,0] += mask[center_c-50:center_c+50:2, ...] * 128

            image_orig_c = np.transpose(image_orig_c, (0,2,1,3))
            image_pred_c = np.transpose(image_pred_c, (0,2,1,3))
            image_mask_c = np.transpose(image_mask_c, (0,2,1,3))

            # save
            if (len(args.select_list) == 100):# | (args.ablation != ""):
                
                rep_save = os.path.join(args.dir, 's_vis' + tag, file_index.replace('_data.nii', '/'))
                if not os.path.exists(rep_save):
                    os.makedirs(rep_save)
                        
                image_orig_c = image_orig_c[:, -1::-1,...]
                image_pred_c = image_pred_c[:, -1::-1,...]
                image_mask_c = image_mask_c[:, -1::-1,...]
                        
                for i, img in enumerate(image_orig):
                    # Image.fromarray(image_orig[i].astype('uint8')).save(rep_save + 'a%d_'%i + file_index.replace('.nii','.png'))
                    Image.fromarray(image_mask[i].astype('uint8')).save(rep_save + 'a%d_'%i + file_index.replace('.nii','_gt.png'))
                    Image.fromarray(image_pred[i].astype('uint8')).save(rep_save + 'a%d_'%i + file_index.replace('.nii','_pred.png'))
                
                    if args.ablation == 'ci':
                        Image.fromarray(image_ci_mask[i].astype('uint8')).save(rep_save + 'a%d_'%i + file_index.replace('.nii','_pred_ci.png'))
                

                if args.ablation != 'ci':                        
                    
                    for i, img in enumerate(image_orig_c):
                        Image.fromarray(image_orig_c[i].astype('uint8')).save(rep_save + 'c%d_'%i + file_index.replace('.nii','_c.png'))
                        Image.fromarray(image_mask_c[i].astype('uint8')).save(rep_save + 'c%d_'%i + file_index.replace('.nii','_c_gt.png'))
                        Image.fromarray(image_pred_c[i].astype('uint8')).save(rep_save + 'c%d_'%i + file_index.replace('.nii','_c_pred.png'))
                        
            else:

                img = montage(image_orig, grid_shape=(8, 5), channel_axis=3)
                msk = montage(image_mask, grid_shape=(8, 5), channel_axis=3)
                prd = montage(image_pred, grid_shape=(8, 5), channel_axis=3)
                
                img_c = montage(image_orig_c, grid_shape=(10, 5), channel_axis=3)
                img_c = img_c[-1::-1,...]
                msk_c = montage(image_mask_c, grid_shape=(10, 5), channel_axis=3)
                msk_c = msk_c[-1::-1,...]
                prd_c = montage(image_pred_c, grid_shape=(10, 5), channel_axis=3)
                prd_c = prd_c[-1::-1,...]

                rep_save = os.path.join(args.dir, 'vis' + tag + '/result_')
                
                Image.fromarray(img.astype('uint8')).save(rep_save + file_index.replace('.nii','.png'))
                Image.fromarray(msk.astype('uint8')).save(rep_save + file_index.replace('.nii','_gt.png'))
                Image.fromarray(prd.astype('uint8')).save(rep_save + file_index.replace('.nii','_pred%s.png'%args.ablation))

                if args.ablation == 'ci':
                    prd_ci = montage(image_ci_mask, grid_shape=(8, 5), channel_axis=3)
                    Image.fromarray(prd_ci.astype('uint8')).save(rep_save + file_index.replace('.nii','_%s.png'%args.ablation))

                else:  
                    Image.fromarray(img_c.astype('uint8')).save(rep_save + file_index.replace('.nii','_c.png'))
                    Image.fromarray(msk_c.astype('uint8')).save(rep_save + file_index.replace('.nii','_c_gt.png'))
                    Image.fromarray(prd_c.astype('uint8')).save(rep_save + file_index.replace('.nii','_c_pred%s.png'%args.ablation))

if __name__ == "__main__":
    main()
