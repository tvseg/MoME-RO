GPUS=1
GPU_DEVICE=0

##### 1_2. GTV Inference
WORKDIR_GTV='/home/gpuadmin/yujin/ro-llama/work_dir/PC_NC/'
CHECK_GTV='v11.0_HU_gtv' 
# CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --target 1 --pretrained_dir $WORKDIR_GTV$CHECK_GTV --test_mode 1 --flag_pc True --save_interval 1
# CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --target 1 --pretrained_dir $WORKDIR_GTV$CHECK_GTV --test_mode 2 --flag_pc True --save_interval 1
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --target 1 --pretrained_dir $WORKDIR_GTV$CHECK_GTV --test_mode 3 --flag_pc True --save_interval 1
# CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --target 1 --pretrained_dir $WORKDIR_GTV$CHECK_GTV --test_mode 0 --flag_pc True --meta True --save_interval 1

# ##### 2_2. CTV Inference
# WORKDIR='/home/gpuadmin/yujin/ro-llama/work_dir/PC_TMI/'
# CHECKPOINTLIST='v17.0_VisionOnly_Stage1all_1.00' 
# for CHECK in $CHECKPOINTLIST
# do
#     echo -E $CHECK
#     if [[ $CHECK == *"llama3"* ]] ; then 
#         # CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --target 2 --gtv_dir $WORKDIR_GTV$CHECK_GTV --pretrained_dir $WORKDIR$CHECK --context True --n_prompts 1 --context_length 0 --test_mode 1 --flag_pc True --textencoder 'llama3' #--save_interval 1 #--textencoder 'llama2_13b'
#         # CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --target 2 --gtv_dir $WORKDIR_GTV$CHECK_GTV --pretrained_dir $WORKDIR$CHECK --context True --n_prompts 1 --context_length 0 --test_mode 2 --flag_pc True --textencoder 'llama3' #--save_interval 1 #--textencoder 'llama2_13b'
#         CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --target 2 --gtv_dir $WORKDIR_GTV$CHECK_GTV --pretrained_dir $WORKDIR$CHECK --context True --n_prompts 1 --context_length 0 --test_mode 3 --flag_pc True --textencoder 'llama3' #--save_interval 1 
#     else
#         # CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --target 2 --gtv_dir $WORKDIR_GTV$CHECK_GTV--pretrained_dir $WORKDIR$CHECK --test_mode 1 --flag_pc True #--save_interval 1
#         # CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --target 2 --gtv_dir $WORKDIR_GTV$CHECK_GTV --pretrained_dir $WORKDIR$CHECK --test_mode 2 --flag_pc True #--save_interval 1
#         CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py --target 2 --gtv_dir $WORKDIR_GTV$CHECK_GTV --pretrained_dir $WORKDIR$CHECK --test_mode 3 --flag_pc True #--save_interval 1
#     fi
#     # CUDA_VISIBLE_DEVICES=$GPU_DEVICE python viewer.py --dir $WORKDIR$CHECK 
# done



