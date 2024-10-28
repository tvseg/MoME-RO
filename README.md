# Official source code for MoME: Mixture of Multicenter Experts in Multimodal Generative AI for Advanced Radiotherapy Target Delineation

## 1. Environment setting
```
git clone https://github.com/tvseg/MM-LLM-RO.git
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
python -m prerocess --dir_ct '/hdd/raw_data/dir_ct/' --dir_add '/hdd/raw_data/dir_mr/' --dir_save '/hdd/raw_data/save_dir/'
```

## 3. Model checkpoints
```
cd MoMCE-RO/model/llama3
git lfs install
git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
cd ..
cd ..
mkdir ckpt
cd ckpt
mkdir multimodal
mkdir organ
cd MoMCE-RO/ckpt/multimodal
download model_best.pt from https://1drv.ms/u/s!AhwNodepZ41ojZhiXf5aalWIpTNUFA?e=Wecdhr
cd ..
cd MoMCE-RO/ckpt/organ
download model_best.pt from https://1drv.ms/u/s!AhwNodepZ41ojOZfRAm6reu33wlXFA?e=e7lI9P
```

## 4. Inference
```
bash MoMCE-RO/test.sh
```

Will be updated in detail soon! 