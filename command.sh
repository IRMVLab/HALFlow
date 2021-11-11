
python main.py \
    --mode train \
    --dataset ft3d \
    --gpu 0 \
    --model HALFlowNet \
    --data_ft3d_path /tmp/FlyingThings3D_subset_processed_35m \
    --data_kitti_path /tmp/KITTI_processed_occ_final \
    --log_dir HALFlowNet_test_ \
    --num_point 8192 \
    --max_epoch 1510 \
    --learning_rate 0.001 \
    --batch_size 8 \
    > log_HALFlowNet.txt 2>&1 &

