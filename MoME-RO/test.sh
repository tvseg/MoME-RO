GPUS=1
GPU_DEVICE=0

##### 1_2. GTV Inference
WORKDIR_GTV='/Users/yo084/Documents/Projects/mnt/0_dataset/MoME/ckpt/'
CHECK_GTV='organ' 
# CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --target 1 --pretrained_dir $WORKDIR_GTV$CHECK_GTV --test_mode 3 --flag_pc True --save_interval 1

##### 2_2. CTV Inference
WORKDIR='/Users/yo084/Documents/Projects/mnt/0_dataset/MoME/ckpt/'
CHECKPOINTLIST='multimodal_MoME' 
for CHECK in $CHECKPOINTLIST
do
    echo -E $CHECK
    if [[ $CHECK == *"multimodal"* ]] ; then 
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --target 2 --gtv_dir $WORKDIR_GTV$CHECK_GTV --pretrained_dir $WORKDIR$CHECK --context True --n_prompts 1 --context_length 32 --test_mode 1 --flag_pc True --textencoder 'llama3' 
        # CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --target 2 --gtv_dir $WORKDIR_GTV$CHECK_GTV --pretrained_dir $WORKDIR$CHECK --context True --n_prompts 1 --context_length 32 --test_mode 2 --flag_pc True --textencoder 'llama3' 
        # CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --target 2 --gtv_dir $WORKDIR_GTV$CHECK_GTV --pretrained_dir $WORKDIR$CHECK --context True --n_prompts 1 --context_length 32 --test_mode 3 --flag_pc True --textencoder 'llama3' 
    else
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --target 2 --gtv_dir $WORKDIR_GTV$CHECK_GTV --pretrained_dir $WORKDIR$CHECK --test_mode 1 --flag_pc True 
    fi
done



