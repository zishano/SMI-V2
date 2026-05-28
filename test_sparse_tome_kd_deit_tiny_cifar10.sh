python test_kt_tome.py --model deit_tiny_16_cifar10 \
    --prune_it -1 \
    --prune_ratio 0 \
    --dataset /home/user/limengkui/data/cifar-10-python \
    --datapool ./output \
    --gpu 0 \
    --ldr 0 \
    --ratio 0 0 0 0 0 0 0 0 0 0 0 0 \
    --model_path /home/user/limengkui/SMI/checkpoints_rfmi
    


# --ratio 0 0 0 0 0 0 0 0 0 0 0 0
# --ratio 1 1 1 1 1 1 1 1 1 1 1 1
# --ratio 2 1 2 1 2 1 2 1 2 1 2 1

# python test_kt_tome.py --model deit_tiny_16_cifar100 \
#     --prune_it 50 100 200 300 \
#     --prune_ratio 0.3 0.3 0.3 0.3 \
#     --dataset /home/user/limengkui/data/cifar-100-python \
#     --datapool ./output \
#     --gpu 0 \
#     --ldr 0
# python test_kt_tome.py --model deit_base_16_cifar10 --prune_it 50 100 200 300 --prune_ratio 0.3 0.3 0.3 0.3 --dataset ../data/cifar/cifar-10-python --datapool ./output --gpu 0
