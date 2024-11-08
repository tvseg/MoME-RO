# Official source code for MoME: Mixture of Multicenter Experts in Multimodal Generative AI for Advanced Radiotherapy Target Delineation (https://arxiv.org/abs/2410.00046)
![alt text](https://github.com/tvseg/MoME-RO/blob/main/Picture1.png) 
![alt text](https://github.com/tvseg/MoME-RO/blob/main/Picture2.png) 

## 1. Environment setting
```
git clone https://github.com/tvseg/MoME-RO.git
pip install -r requirements.txt
```

## 2. Dataset
```
# Prepare raw data following below structure
CT data : PatientNumber_CT~/*.dcm
Label data : PatientNumber_RTst~/*.dcm

# Examples (PatientNumber = Sample1 or Sample2)
/hdd/raw_data/dir_ct/
    └── 2024-01__Studies
        ├── Sample1_CT_2024-01-11_111051_._Body.3.0_n145__00000
        ├── Sample1_RTst_2024-01-11_111051_._Prostate.1st.simplified_n1__00000
        ├── 1_CT_2024-01-02_114118_._Body.3.0_n170__00000
            └── 1.2.392.200036.9116.2.6.1.1236.3294313065.1704163499.830744.dcm
                                            ...
            └── 1.2.392.200036.9116.2.6.1.1236.3294313065.1704163500.38142.dcm
        └── 1_RTst_2024-01-02_114118_._Target+OAR.MSC.simplified_n1__00000
            └── 2.16.840.1.114362.1.12105090.23690985581.662027519.843.3197.dcm

# Run preprocess.py as below
python -m preprocess --dir_ct '/hdd/raw_data/dir_ct/' --dir_add '/hdd/raw_data/dir_mr/' --dir_save '/hdd/raw_data/save_dir/'
```

## 3. Model checkpoints
```
cd MoME-RO/model/llama3
git lfs install
git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
cd ..
cd ..
mkdir ckpt
cd ckpt
mkdir multimodal_MoME
mkdir organ
cd MoME-RO/ckpt/multimodal_MoME
download model_best.pt from https://1drv.ms/u/s!AhwNodepZ41ojZhiXf5aalWIpTNUFA?e=Wecdhr
cd ..
cd MoME-RO/ckpt/organ
download model_best.pt from https://1drv.ms/u/s!AhwNodepZ41ojOZfRAm6reu33wlXFA?e=e7lI9P
```

## 4. Hyperparameter settings
```
set hyperparameters in main.py to each directory
- data_dir : /hdd/raw_data/save_dir/ (example anonymized data will be updated)
- report_dir : /hdd/raw_data/save_dir/report.xlsx (example anonymized report format will be updated)
- rep_llm : MoME-RO/model/llama3/Meta-Llama-3-8B-Instruct
- gtv_dir : MoME-RO/ckpt/organ
- pretrained_dir : MoME-RO/ckpt/multimodal_MoME
- logdir : MoME-RO/run/multimodal_MoME_Finetune (if finetune)
```

## 5. Inference
```
bash MoME-RO/test.sh
```

## 6. Finetune
```
set hyperparameter "force_expert" of finetune.sh for multimodal_MoME based on different experts
- force_expert : 1 (if metrics of MoME-RO/ckpt/multimodal_MoME/result_centerD-expert1.xlsx shows the best results)
- force_expert : 2 (if metrics of MoME-RO/ckpt/multimodal_MoME/result_centerD-expert2.xlsx shows the best results)
- force_expert : 3 (if metrics of MoME-RO/ckpt/multimodal_MoME/result_centerD-expert3.xlsx shows the best results)
set hyperparameter "shot" of finetune.sh  from 1 to 3 for each 1-shot to 3-shots
- shot : 3
set hyperparameter "report_dir" of finetune.sh between "model_best.pt" or "model_last.pt" depending on utilizing early-stopping or not
- report_dir : model_last.pt
bash MoME-RO/finetune.sh
```
