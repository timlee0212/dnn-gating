# Training
python3.9 imagenet.py --wbits 8 --abits 8 --gpu 3 --save --arch levit_256 --epochs 300 --dense --pretrained --resize 224 --lr 1e-5 --test --resume remote
# Inference
#python3.9 imagenet.py --wbits 8 --abits 8 --gpu 3 --save --arch deit_small_patch16_224 --epochs 300 --dense --pretrained --resize 224 --lr 1e-5 --test 

#python3.9 cifar10.py --wbits 8 --abits 8 --gpu 3 --save --arch deit_tiny_patch16_224 --epochs 300 --dense --pretrained --resize 224 --lr 1e-5 --test
