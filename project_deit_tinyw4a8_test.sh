# python test_quant_tome.py --model deit_tiny_16_imagenet \
#     --prune_it 50 100 200 300 \
#     --prune_ratio 0.3 0.3 0.3 0.3 \
#     --dataset ../project/data/imagenet \
#     --datapool ./output \
#     --mode 0 \
#     --w_bit 4 --a_bit 8 \
#     --calib-batchsize 32 \
#     --val-batchsize 50 \
#     --gpu 0 \
#     --ldr 0 \
#     --ratio 0 0 0 0 0 0 0 0 0 0 0 0

python test_quant_tome.py --model deit_tiny_16_imagenet \
    --prune_it 50 100 200 300 \
    --prune_ratio 0.3 0.3 0.3 0.3 \
    --dataset ../project/data/imagenet \
    --datapool ./output \
    --mode 0 \
    --w_bit 4 --a_bit 8 \
    --calib-batchsize 32 \
    --val-batchsize 50 \
    --gpu 0 \
    --ldr 0.52 \
    --ratio 1 1 1 1 1 1 1 1 1 1 1 1

