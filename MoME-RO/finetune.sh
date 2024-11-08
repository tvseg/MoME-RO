GPUS=1
GPU_DEVICE=0

##### 1_3. GTV 
WORKDIR_GTV='/Users/yo084/Documents/Projects/mnt/0_dataset/MoME/ckpt/'
CHECK_GTV='organ' 

##### 2_3. CTV Finetune
WORKDIR='/Users/yo084/Documents/Projects/mnt/0_dataset/MoME/ckpt/'
SAVEDIR='/Users/yo084/Documents/Projects/mnt/3_output/MoME/ckpt/'
CHECKPOINTLIST='unimodal_Finetune_3shot multimodal_MoME_Finetune_3shot' 
for CHECK in $CHECKPOINTLIST
do
    echo -E $CHECK
    if [[ $CHECK == *"multimodal"* ]] ; then 
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --target 2 --stage 3 --gtv_dir $WORKDIR_GTV$CHECK_GTV --pretrained_dir $WORKDIR$CHECK --logdir $SAVEDIR$CHECK --context True --n_prompts 1 --context_length 32 --test_mode 4 --force_expert 2 --flag_pc True --textencoder 'llama3' --shot 3
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --target 2 --gtv_dir $WORKDIR_GTV$CHECK_GTV --pretrained_dir $WORKDIR$CHECK --context True --n_prompts 1 --context_length 32 --test_mode 4 --force_expert 2 --flag_pc True --textencoder 'llama3' --pretrained_model_name 'model_last.pt'
    else
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --target 2 --stage 3 --gtv_dir $WORKDIR_GTV$CHECK_GTV --pretrained_dir $WORKDIR$CHECK --logdir $SAVEDIR$CHECK --test_mode 4 --flag_pc True --shot 3
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --target 2 --gtv_dir $WORKDIR_GTV$CHECK_GTV --pretrained_dir $WORKDIR$CHECK --test_mode 4 --flag_pc True --pretrained_model_name 'model_last.pt'
    fi
done



