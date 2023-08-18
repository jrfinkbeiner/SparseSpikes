for USE_AUG in 1 0
    do
    for BATCHSIZE in 256 48
        do
        for NUM_HIDDEN_LAYERS in 3 4 5 6
        do
            for MAX_ACT in 0.01 0.05 0.1
            do
                CUDA_VISIBLE_DEVICES=2 python3 train_lif.py --dataset_name=SHD --lr=3e-3 --use_sparse=1 --second_threshold=0.9 --use_wand=1 --batchsize=$BATCHSIZE --batchsize_test=1132 --use_thresh_scheduler=1 --thresh_scheduler_mul=0.5 --ro_type=linear_ro --sparse_size_inp=48 --max_activity=$MAX_ACT --num_epochs=150 --seq_len=500 --num_hidden_layers=$NUM_HIDDEN_LAYERS --use_aug=$USE_AUG --use_crop=0 --act_rate=0.05 --use_lr_scheduler=1
            done
            CUDA_VISIBLE_DEVICES=2 python3 train_lif.py --dataset_name=SHD --lr=3e-3 --use_sparse=0 --second_threshold=0.9 --use_wand=1 --batchsize=$BATCHSIZE --batchsize_test=1132 --use_thresh_scheduler=1 --thresh_scheduler_mul=0.5 --ro_type=linear_ro --sparse_size_inp=48 --max_activity=$MAX_ACT --num_epochs=150 --seq_len=500 --num_hidden_layers=$NUM_HIDDEN_LAYERS --use_aug=$USE_AUG --use_crop=0 --act_rate=0.05 --use_lr_scheduler=1
        done
    done
done