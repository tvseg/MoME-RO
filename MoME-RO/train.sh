GPUS=1
GPU_DEVICE=0

# ##### 1_1.GTV Segmentation
WORKDIR_GTV='/Users/yo084/Documents/Projects/mnt/0_dataset/MoME/ckpt'
CHECK_GTV='organ' 

##### 2_1. CTV Segmentation
WORKDIR='/Users/yo084/Documents/Projects/mnt/0_dataset/MoME/ckpt'
CHECKPOINTLIST='multimodal' 
for CHECK in $CHECKPOINTLIST
do
    echo -E $CHECK
    if [[ $CHECK == *"multimodal"* ]] ; then 
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --target 2 --gtv_dir $WORKDIR_GTV$CHECK_GTV --logdir $WORKDIR$CHECK --context True --n_prompts 1 --context_length 32 --flag_pc True --textencoder 'llama3'
    else
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --target 2 --stage 3 --gtv_dir $WORKDIR_GTV$CHECK_GTV --logdir $WORKDIR$CHECK --flag_pc True
    fi
done




