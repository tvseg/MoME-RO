import os
import numpy as np
import logging

import re
import glob
import argparse

import SimpleITK as sitk
from dcmrtstruct2nii.adapters.convert.rtstructcontour2mask import DcmPatientCoords2Mask
from dcmrtstruct2nii.adapters.convert.filenameconverter import FilenameConverter
from dcmrtstruct2nii.adapters.input.contours.rtstructinputadapter import RtStructInputAdapter
from dcmrtstruct2nii.adapters.input.image.dcminputadapter import DcmInputAdapter
from dcmrtstruct2nii.adapters.output.niioutputadapter import NiiOutputAdapter
from dcmrtstruct2nii.exceptions import PathDoesNotExistException, ContourOutOfBoundsException
from dcmrtstruct2nii.exceptions import InvalidFileFormatException
from dcmrtstruct2nii.adapters.input.abstractinputadapter import AbstractInputAdapter

from tqdm import tqdm
import pydicom
from PIL import Image
from skimage.util import montage 

FLAG_SAVE = True


def dcmrtstruct2nii(rtstruct_file, dicom_file, output_path, additional_dicom_file=None, structures=None, gzip=True, mask_background_value=0, mask_foreground_value=255, convert_original_dicom=True, series_id=None):  # noqa: C901 E501
    """
    Converts A DICOM and DICOM RT Struct file to nii

    :param rtstruct_file: Path to the rtstruct file
    :param dicom_file: Path to the dicom file
    :param output_path: Output path where the masks are written to
    :param structures: Optional, list of structures to convert
    :param gzip: Optional, output .nii.gz if set to True, default: True
    :param series_id: Optional, the Series Instance UID. Use  to specify the ID corresponding to the image if there are
    dicoms from more than one series in `dicom_file` folder

    :raise InvalidFileFormatException: Raised when an invalid file format is given.
    :raise PathDoesNotExistException: Raised when the given path does not exist.
    :raise UnsupportedTypeException: Raised when conversion is not supported.
    :raise ValueError: Raised when mask_background_value or mask_foreground_value is invalid.
    """
    output_path = os.path.join(output_path, '')  # make sure trailing slash is there

    if not os.path.exists(rtstruct_file):
        raise PathDoesNotExistException(f'rtstruct path does not exist: {rtstruct_file}')

    if not os.path.exists(dicom_file):
        raise PathDoesNotExistException(f'DICOM path does not exists: {dicom_file}')

    if mask_background_value < 0 or mask_background_value > 255:
        raise ValueError(f'Invalid value for mask_background_value: {mask_background_value}, must be between 0 and 255')

    if mask_foreground_value < 0 or mask_foreground_value > 255:
        raise ValueError(f'Invalid value for mask_foreground_value: {mask_foreground_value}, must be between 0 and 255')

    if structures is None:
        structures = []

    os.makedirs(output_path, exist_ok=True)

    filename_converter = FilenameConverter()
    rtreader = RtStructInputAdapter()

    rtstructs = rtreader.ingest(rtstruct_file)
    dicom_image = DcmInputAdapter().ingest(dicom_file, series_id=series_id)

    dcm_patient_coords_to_mask = DcmPatientCoords2Mask()
    nii_output_adapter = NiiOutputAdapter()

    if additional_dicom_file is not None:
        logging.info('Converting addtional sequence DICOM to nii')
        additional_dicom_image = DcmInputAdapter().ingest(additional_dicom_file, series_id=series_id)
        nii_output_adapter.write(additional_dicom_image, f'{output_path}image_add_reg', gzip)

    for rtstruct in rtstructs:
        if len(structures) == 0 or rtstruct['name'] in structures:
            if 'sequence' not in rtstruct:
                logging.info('Skipping mask {} no shape/polygon found'.format(rtstruct['name']))
                continue

            logging.info('Working on mask {}'.format(rtstruct['name']))
            try:
                mask = dcm_patient_coords_to_mask.convert(rtstruct['sequence'], dicom_image, mask_background_value, mask_foreground_value)
            except ContourOutOfBoundsException:
                logging.warning(f'Structure {rtstruct["name"]} is out of bounds, ignoring contour!')
                continue

            mask.CopyInformation(dicom_image)

            mask_filename = filename_converter.convert(f'mask_{rtstruct["name"]}')
            nii_output_adapter.write(mask, f'{output_path}{mask_filename}', gzip)

    if convert_original_dicom:
        logging.info('Converting original DICOM to nii')
        nii_output_adapter.write(dicom_image, f'{output_path}image', gzip)

    logging.info('Success!')


def test_save(save_path, image, name, flag_save=FLAG_SAVE):   

    if flag_save:
        img_add_orig = sitk.GetArrayFromImage(image)
        if name.find('1_ct') >= 0:
            img_add_orig = img_add_orig.clip(-200, 250)

        img_add_orig = (img_add_orig - np.min(img_add_orig)) / (np.max(img_add_orig) - np.min(img_add_orig))
        if name.find('3_mr') >= 0:
            # img_add_orig = img_add_orig[img_add_orig.shape[0]//2-60:img_add_orig.shape[0]//2+20:2, ...]
            img_add_orig = img_add_orig[:len(img_add_orig):len(img_add_orig)//40, ...][:40]
        else:
            if len(img_add_orig)>20:
                img_add_orig = img_add_orig[img_add_orig.shape[0]//2-20:img_add_orig.shape[0]//2+20, ...]
                # img_add_orig = img_add_orig[:len(img_add_orig):len(img_add_orig)//40, ...][:40]
        img_add_orig = np.expand_dims(img_add_orig, axis=3).repeat(3,3)

        img = montage(img_add_orig, grid_shape=(8, 5), channel_axis=3)
        Image.fromarray((img*255).astype('uint8')).save('%s/%s.png'%(save_path, name))
                
    else:
        pass


# Function to print DICOM file structure
def print_dicom_structure(dataset, indent=0):
    """Recursively print the DICOM dataset structure."""
    indent_str = "  " * indent
    for elem in dataset:
        if elem.VR == "SQ":  # Check if the element is a Sequence
            print(f"{indent_str}Sequence: {elem.tag}, {elem.name}")
            for item in elem:
                print_dicom_structure(item, indent + 1)
        else:
            print(f"{indent_str}Element: {elem.tag}, {elem.name}, {elem.value}")


# Function to extract transformation parameters from the DICOM file
def parse_dice_reg(dicom_file_path):
    # Read the DICOM file
    dicom_data = pydicom.dcmread(dicom_file_path)
    # print_dicom_structure(dicom_data)

    sequences_to_check = [
            (0x0070, 0x0308)  # Example tag for Spatial Registration Sequence
        ]

    # Navigate through sequences
    current_dataset = dicom_data
    for seq_tag in sequences_to_check:
        if seq_tag in current_dataset:
            sequence = current_dataset[seq_tag]
            for item in sequence:
                if (0x0070, 0x0309) in item:
                    matrix_reg_seq = item[0x0070, 0x0309][0]
                    matrix_seq = matrix_reg_seq[0x0070, 0x030a][0]
                    transformation_matrix = matrix_seq[0x3006, 0x00c6].value
                else:
                    print(f"Transform Parameters Sequence not found in item: {item}")
        else:
            print(f"Sequence tag {seq_tag} not found in the current dataset.")


    # Convert the transformation matrix to a numpy array
    transformation_array = np.array(transformation_matrix).reshape((4, 4))
    print("Transformation Matrix:")
    # print(transformation_array)

    return transformation_array


def main(args):

    # Reference setting
    reference_direction = np.identity(3).flatten()
    reference_spacing = np.array((1., 1., 3)) 

    # Directory
    dirs = glob.glob('%s/**/*_CT_*'%args.dir_ct, recursive=True)
    # add_dirs = args.dir_add
    save_dirs = args.dir_save

    outlier = []
    mask_outlier = []
    add_outlier = []

    dirs.sort()
    for folder in tqdm(dirs):

        # CT
        folder_split = folder.split('_CT')[0].split('/')
        patient_rep = '/'.join(folder_split[:-1])
        print(patient_rep)

        patient_ID = folder_split[-1].split('_')[-1]
        save_path = '%s/%s'%(save_dirs, patient_ID)

        # MRI
        new_add_subfolder = None
        # try:
        #     add_subfolder = glob.glob('%s/**/*%s*MR*'%(add_dirs, patient_ID), recursive=True)[0]
        #     new_add_subfolder = add_subfolder.replace(' ', '_')
        #     os.rename(add_subfolder, new_add_subfolder)     
        # except:
        #     continue

        os.makedirs(save_path, exist_ok=True)

        # Indent removal
        new_subfolder = folder.replace(' ', '_')
        os.rename(folder, new_subfolder)

        # RTST
        rtstruct_path_cand = glob.glob('%s/*%s*RTst*/*.dcm'%(patient_rep, patient_ID), recursive=True)
            
        for rtstruct_path in rtstruct_path_cand:

            # Indent removal
            new_rtstruct_path = rtstruct_path.replace(' ', '_')
            try:
                os.rename('/'.join(rtstruct_path.split('/')[:-1]), '/'.join(new_rtstruct_path.split('/')[:-1]))
            except:
                pass
            
            # Convert DICOM RTSTRUCT to NIfTI
            _ = dcmrtstruct2nii(rtstruct_file=new_rtstruct_path, dicom_file=new_subfolder, output_path=save_path, additional_dicom_file=new_add_subfolder)
                
        # Load NIfTI to resample
        image_add = None
        os.chdir(save_path)
        try:
            image = sitk.ReadImage('image.nii.gz') 
        except:
            outlier.append(patient_ID)
            print("outlier (%d)"%(len(outlier)))
            continue
        # try:
        #     image_add = sitk.ReadImage('image_add_reg.nii.gz')
        # except:
        #     image_add = None
        #     continue

        # Get image specification
        image_size = image.GetSize()
        image_spacing = image.GetSpacing()
        image_origin = image.GetOrigin()
        image_direction = image.GetDirection()
        label = np.zeros((image_size[2], image_size[0], image_size[1]))
        
        # Trial 1: Extract adequate masks
        for j in glob.glob('mask*'):
            if (j.lower().find('tv') >= 0):

                structure = sitk.ReadImage(j)
                npstructure = sitk.GetArrayFromImage(structure)

                # GTV
                if (j.lower().find('gtv') >= 0) | (j.lower().find('_250x') >= 0) | (j.lower().find('_240') >= 0) | (j.lower().find('hptv') >= 0) | (j.lower().find('ptv1') >= 0)| (j.lower().find('high') >= 0):
                    label[npstructure == 255] = 2 #label_index[j.lower()] #
                    print('***gtv***'+j, sep = ', ')

                find = j.lower().split('tv')[-1]
                num_find = re.findall(r'\d+', find)
                if len(num_find) == 0:
                    print(j, sep = ', ')
                    continue

                # CTV & PTV
                label[(npstructure == 255) & (label!= 2)] = 1
                print('***'+j, sep = ', ')

            else:
                print(j, sep = ', ')

            os.remove(j)
        
        # Set reference
        new_size = np.array(image_size) * np.array(image_spacing) / reference_spacing
        reference_size = [int(new_size[0]), int(new_size[1]), int(new_size[2])]
        reference_image = sitk.Image(reference_size, sitk.sitkFloat32)
        reference_image.SetOrigin(image_origin)
        reference_image.SetSpacing(reference_spacing)
        reference_image.SetDirection(reference_direction)

        # Set interpolator 
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference_image)
        resampler.SetOutputOrigin(reference_image.GetOrigin())

        # Set resampler
        resampler.SetOutputSpacing(reference_image.GetSpacing())
        resampler.SetOutputDirection(reference_image.GetDirection())
        resampler.SetInterpolator(sitk.sitkLinear)

        # Set interpolator for CT data
        out_img = resampler.Execute(image)
        test_save(save_path, out_img, '1_ct')
        sitk.WriteImage(out_img, '%s/data.nii.gz'%save_path)

        # Set interpolator for MR data
        if image_add is not None:
            
            # Origin change
            test_save(save_path, image_add, '3_mr_before') # 3_mr_before_reg

            # Resample following CT resampling
            out_im_add2 = resampler.Execute(image_add)
            test_save(save_path, out_im_add2, '2_mr_reg', True)
            # sitk.WriteImage(out_im_add2, '%s/data_add_reg.nii.gz'%save_path)
       
        # Set interpolator for label
        itkLabel = sitk.GetImageFromArray(label)
        itkLabel.SetSpacing(image_spacing)
        itkLabel.SetOrigin(image_origin)
        itkLabel.SetDirection(image_direction)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        out_label = resampler.Execute(itkLabel)

        test_save(save_path, out_label, '4_label')

        # # label_orig = sitk.GetArrayFromImage(out_label)
        sitk.WriteImage(out_label, 'label_gtv.nii.gz')

    print("outlier: (%d): %s"%(len(outlier), ', '.join(outlier)))
    print("add outlier (%d): %s"%(len(add_outlier), ', '.join(add_outlier)))
    print("mask outlier (%d): %s"%(len(mask_outlier), ', '.join(mask_outlier)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ContextSeg")    
    parser.add_argument("--dir_ct", default='/Users/yo084/Documents/Projects/mnt/0_dataset/MoME/MGH/dir_ct/', type=str, help="root directory for ct sequence")
    parser.add_argument("--dir_add", default='', type=str, help="root directory for additional sequence (MR, PET, SPECT)") 
    parser.add_argument("--dir_save", default='/Users/yo084/Documents/Projects/mnt/0_dataset/MoME/center/centerD/', type=str, help="target directory for processed dataset") 
    args = parser.parse_args()

    main(args)
    