

for NUM_HIDDEN_LAYERS in 3 2
    do
    for BATCHSIZE in 256 48
    do
        for USE_AUG in 0
        do
            for USE_LR_SCHDULER in 1
            do

		for MAX_ACT in 0.01 0.05 0.1
		do
                CUDA_VISIBLE_DEVICES=2 python3 train_lif.py --dataset_name=NMNIST --lr=1e-2 --use_lr_scheduler=$USE_LR_SCHDULER --use_sparse=1 --second_threshold=0.9 --use_wand=1 --batchsize=$BATCHSIZE --batchsize_test=1000 --use_thresh_scheduler=1 --thresh_scheduler_mul=0.5 --ro_type=linear_ro --sparse_size_inp=32 --max_activity=$MAX_ACT --num_epochs=50 --seq_len=250 --num_hidden_layers=$NUM_HIDDEN_LAYERS --use_aug=$USE_AUG --use_crop=1 --act_rate=0.05
		done
            done
        done
    done
done

# exit



# for NUM_HIDDEN_LAYERS in 3 2
#     do
#     for BATCHSIZE in 256 48
#     do
#         for USE_AUG in 0
#         do
#             for USE_LR_SCHDULER in 1
#             do
#                 CUDA_VISIBLE_DEVICES=3 python3 train_lif.py --dataset_name=NMNIST --lr=1e-2 --use_lr_scheduler=$USE_LR_SCHDULER --use_sparse=0 --second_threshold=-100. --use_wand=1 --batchsize=$BATCHSIZE --batchsize_test=1000 --use_thresh_scheduler=0 --thresh_scheduler_mul=0.5 --ro_type=linear_ro --sparse_size_inp=32 --max_activity=0.05 --num_epochs=50 --seq_len=250 --num_hidden_layers=$NUM_HIDDEN_LAYERS --use_aug=$USE_AUG --use_crop=1 --act_rate=0.05
#             done
#         done
#     done
# done





