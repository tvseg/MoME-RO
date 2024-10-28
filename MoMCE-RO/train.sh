GPUS=1
GPU_DEVICE=0

# ##### 1_1.GTV Segmentation
WORKDIR_GTV='/Users/yo084/Documents/Projects/1_MoMCE-RO_vFinal/MoME-RO/MoMCE-RO/ckpt'
CHECK_GTV='organ' 

##### 2_1. CTV Segmentation
WORKDIR='/Users/yo084/Documents/Projects/1_MoMCE-RO_vFinal/MoME-RO/MoMCE-RO/ckpt'
CHECKPOINTLIST='multimodal' 
for CHECK in $CHECKPOINTLIST
do
    echo -E $CHECK
    if [[ $CHECK == *"multimodal"* ]] ; then 
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --target 2 --gtv_dir $WORKDIR_GTV$CHECK_GTV --logdir $WORKDIR$CHECK --context True --n_prompts 1 --context_length 0 --flag_pc True --textencoder 'llama3'
    elif [[ $CHECK == *"t5"* ]] ; then 
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --target 2 --gtv_dir $WORKDIR_GTV$CHECK_GTV --logdir $WORKDIR$CHECK --context True --n_prompts 2  --context_length 0 --context_mode 1 --compare_mode 1 --textencoder 't5'
    else
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --target 2 --stage 3 --gtv_dir $WORKDIR_GTV$CHECK_GTV --logdir $WORKDIR$CHECK --flag_pc True
    fi
done




