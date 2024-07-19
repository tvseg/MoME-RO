GPUS=1
GPU_DEVICE=0

# ##### 1_1.GTV Segmentation
WORKDIR_GTV='/home/gpuadmin/yujin/ro-llama/work_dir/PC_NC/'
CHECK_GTV='v11.0_HU_gtv' 
# # CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --target 1 --logdir $WORKDIR_GTV$CHECK_GTV --flag_pc True

##### 2_1. CTV Segmentation
WORKDIR='/home/gpuadmin/yujin/ro-llama/work_dir/PC_TMI/'
CHECKPOINTLIST='v17.0_VisionOnly_Stage1all_1.00' 
for CHECK in $CHECKPOINTLIST
do
    echo -E $CHECK
    if [[ $CHECK == *"llama3"* ]] ; then 
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --target 2 --gtv_dir $WORKDIR_GTV$CHECK_GTV --logdir $WORKDIR$CHECK --context True --n_prompts 1 --context_length 0 --flag_pc True --textencoder 'llama3' #--checkpoint '/home/gpuadmin/yujin/ro-llama/work_dir/PC_NC/v10.1_llama2_finegrained_p4_n1_b2_1.00/model_last.pt' #--textencoder 'llama2_13b'
    elif [[ $CHECK == *"t5"* ]] ; then 
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --target 2 --gtv_dir $WORKDIR_GTV$CHECK_GTV --logdir $WORKDIR$CHECK --context True --n_prompts 2  --context_length 0 --context_mode 1 --compare_mode 1 --textencoder 't5'
    else
        CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --target 2 --gtv_dir $WORKDIR_GTV$CHECK_GTV--logdir $WORKDIR$CHECK --flag_pc True
    fi
done




