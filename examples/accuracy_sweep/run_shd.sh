# CUDA_VISIBLE_DEVICES=3 python3 train_lif.py --dataset_name=DVSGesture --lr=3e-3 --use_sparse=0 --second_threshold=-100 --use_wand=1 --batchsize=256 --batchsize_test=264 --use_thresh_scheduler=0 --thresh_scheduler_mul=0.01 --ro_type=linear_ro --sparse_size_inp=96 --max_activity=.05 --num_epochs=50 --seq_len=100 --num_hidden_layers=2 --use_aug=1 --act_rate=0.05
# # CUDA_VISIBLE_DEVICES=3 python3 train_lif.py --dataset_name=DVSGesture --lr=3e-3 --use_sparse=0 --second_threshold=-100 --use_wand=1 --batchsize=256 --batchsize_test=264 --use_thresh_scheduler=0 --thresh_scheduler_mul=0.01 --ro_type=linear_ro --sparse_size_inp=96 --max_activity=.05 --num_epochs=50 --seq_len=250 --num_hidden_layers=2 --use_aug=1 --act_rate=0.05
# CUDA_VISIBLE_DEVICES=3 python3 train_lif.py --dataset_name=DVSGesture --lr=3e-3 --use_sparse=0 --second_threshold=-100 --use_wand=1 --batchsize=256 --batchsize_test=264 --use_thresh_scheduler=0 --thresh_scheduler_mul=0.01 --ro_type=linear_ro --sparse_size_inp=96 --max_activity=.05 --num_epochs=50 --seq_len=400 --num_hidden_layers=2 --use_aug=1 --act_rate=0.05
# # CUDA_VISIBLE_DEVICES=3 python3 train_lif.py --dataset_name=DVSGesture --lr=3e-3 --use_sparse=0 --second_threshold=-100 --use_wand=1 --batchsize=256 --batchsize_test=264 --use_thresh_scheduler=0 --thresh_scheduler_mul=0.01 --ro_type=linear_ro --sparse_size_inp=96 --max_activity=.05 --num_epochs=50 --seq_len=1000 --num_hidden_layers=2 --use_aug=1 --act_rate=0.05

# CUDA_VISIBLE_DEVICES=0 python3 train_lif.py --dataset_name=DVSGesture --lr=3e-3 --use_sparse=0 --second_threshold=-100 --use_wand=1 --batchsize=256 --batchsize_test=264 --use_thresh_scheduler=0 --thresh_scheduler_mul=0.01 --ro_type=linear_ro --sparse_size_inp=96 --max_activity=.05 --num_epochs=50 --seq_len=100 --num_hidden_layers=2 --use_aug=1 --act_rate=0.05 &> log_0.log &
# CUDA_VISIBLE_DEVICES=0 python3 train_lif.py --dataset_name=DVSGesture --lr=3e-3 --use_sparse=0 --second_threshold=-100 --use_wand=1 --batchsize=256 --batchsize_test=264 --use_thresh_scheduler=0 --thresh_scheduler_mul=0.01 --ro_type=linear_ro --sparse_size_inp=96 --max_activity=.05 --num_epochs=50 --seq_len=400 --num_hidden_layers=2 --use_aug=1 --act_rate=0.05 &> log_1.log &
# CUDA_VISIBLE_DEVICES=0 python3 train_lif.py --dataset_name=DVSGesture --lr=3e-3 --use_sparse=0 --second_threshold=-100 --use_wand=1 --batchsize=256 --batchsize_test=264 --use_thresh_scheduler=0 --thresh_scheduler_mul=0.01 --ro_type=linear_ro --sparse_size_inp=96 --max_activity=.05 --num_epochs=50 --seq_len=100 --num_hidden_layers=3 --use_aug=1 --act_rate=0.05 &> log_2.log &
# CUDA_VISIBLE_DEVICES=0 python3 train_lif.py --dataset_name=DVSGesture --lr=3e-3 --use_sparse=0 --second_threshold=-100 --use_wand=1 --batchsize=256 --batchsize_test=264 --use_thresh_scheduler=0 --thresh_scheduler_mul=0.01 --ro_type=linear_ro --sparse_size_inp=96 --max_activity=.05 --num_epochs=50 --seq_len=400 --num_hidden_layers=3 --use_aug=1 --act_rate=0.05 &> log_3.log &



# CUDA_VISIBLE_DEVICES=1 python3 train_lif.py --dataset_name=DVSGesture --lr=3e-3 --use_sparse=0 --second_threshold=-100 --use_wand=1 --batchsize=256 --batchsize_test=264 --use_thresh_scheduler=0 --thresh_scheduler_mul=0.01 --ro_type=linear_ro --sparse_size_inp=96 --max_activity=.05 --num_epochs=50 --seq_len=1000 --num_hidden_layers=3 --use_aug=1 --use_crop=1 --act_rate=0.05


# CUDA_VISIBLE_DEVICES=3 python3 train_lif.py --dataset_name=DVSGesture --lr=3e-3 --use_sparse=0 --second_threshold=0.95 --use_wand=1 --batchsize=256 --batchsize_test=264 --use_thresh_scheduler=1 --thresh_scheduler_mul=0.1 --ro_type=linear_ro --sparse_size_inp=96 --max_activity=.05 --num_epochs=50 --seq_len=1000 --num_hidden_layers=2 --use_aug=1 --act_rate=0.05


# for SEQ_LEN in 100 500
# do
#     for NUM_HIDDEN_LAYERS in 2 3 4 
#     do
#         CUDA_VISIBLE_DEVICES=1 python3 train_lif.py --dataset_name=SHD --lr=3e-3 --use_sparse=0 --second_threshold=-100 --use_wand=1 --batchsize=256 --batchsize_test=2264 --use_thresh_scheduler=1 --thresh_scheduler_mul=0.5 --ro_type=linear_ro --sparse_size_inp=48 --max_activity=0.05 --num_epochs=100 --seq_len=$SEQ_LEN --num_hidden_layers=$NUM_HIDDEN_LAYERS --use_aug=1 --use_crop=0 --act_rate=0.05
#     done
# done




# for NUM_HIDDEN_LAYERS in 3 4 5 6
# do
#     for THRESH_MUL in 0.5
#     do
#         for MAX_ACT in 0.01 0.025 0.05 0.1 0.25
#         do
#             CUDA_VISIBLE_DEVICES=2 python3 train_lif.py --dataset_name=SHD --lr=3e-3 --use_sparse=1 --second_threshold=0.9 --use_wand=1 --batchsize=256 --batchsize_test=2264 --use_thresh_scheduler=1 --thresh_scheduler_mul=$THRESH_MUL --ro_type=linear_ro --sparse_size_inp=48 --max_activity=$MAX_ACT --num_epochs=100 --seq_len=500 --num_hidden_layers=$NUM_HIDDEN_LAYERS --use_aug=1 --use_crop=0 --act_rate=0.05
#         done
#     done
# done



for USE_AUG in 1 0
    do
    for BATCHSIZE in 256 48
        do
        for NUM_HIDDEN_LAYERS in 3 4 5 6
        do
            # CUDA_VISIBLE_DEVICES=0 python3 train_lif.py --dataset_name=SHD --lr=3e-3 --use_sparse=0 --second_threshold=-100 --use_wand=1 --batchsize=$BATCHSIZE --batchsize_test=1132 --use_thresh_scheduler=0 --thresh_scheduler_mul=0.5 --ro_type=linear_ro --sparse_size_inp=48 --max_activity=0.05 --num_epochs=150 --seq_len=500 --num_hidden_layers=$NUM_HIDDEN_LAYERS --use_aug=$USE_AUG --use_crop=0 --act_rate=0.05 --use_lr_scheduler=1
            CUDA_VISIBLE_DEVICES=3 python3 train_lif.py --dataset_name=SHD --lr=3e-3 --use_sparse=1 --second_threshold=0.9 --use_wand=1 --batchsize=$BATCHSIZE --batchsize_test=1132 --use_thresh_scheduler=1 --thresh_scheduler_mul=0.5 --ro_type=linear_ro --sparse_size_inp=48 --max_activity=0.1 --num_epochs=150 --seq_len=500 --num_hidden_layers=$NUM_HIDDEN_LAYERS --use_aug=$USE_AUG --use_crop=0 --act_rate=0.05 --use_lr_scheduler=1
        done
    done
done







# for MAX_ACT in 0.01 0.025 0.05 0.1 0.25
# do
#     CUDA_VISIBLE_DEVICES=1 python3 train_lif.py --dataset_name=SHD --lr=3e-3 --use_sparse=1 --second_threshold=0.95 --use_wand=1 --batchsize=256 --batchsize_test=264 --use_thresh_scheduler=1 --thresh_scheduler_mul=0.5 --ro_type=linear_ro --sparse_size_inp=96 --max_activity=$MAX_ACT --num_epochs=50 --seq_len=1000 --num_hidden_layers=3 --use_aug=1 --use_crop=1 --act_rate=0.05
# done
