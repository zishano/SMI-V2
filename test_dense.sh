python test_quant.py --model deit_base_16_imagenet \
    --prune_it -1 \
    --prune_ratio 0 \
    --dataset ../data/imagenet \
    --datapool ./output \
    --mode 0 \
    --w_bit 8 --a_bit 8 \
    --calib-batchsize 8 \
    --val-batchsize 50 \
    --gpu 3