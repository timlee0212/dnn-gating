#python3.9 cifar10.py -w 32 -a 32 -gtar 1.0 -spbp -pact -gpu 0 -s -lr 0.001 -th 1.0 -ft #91.2% +sp:0
#python3.9 cifar10.py -w 8 -a 8 -gtar 1.0 -spbp -pact -gpu 0 -s -lr 0.001 -th 1.0 -ft -model w32a32 #91.2% +sp:0
#python3.9 cifar10.py -w 4 -a 4 -gtar 1.0 -spbp -pact -gpu 0 -s -lr 0.1 -th 1.0 -ft -model w32a32 #91.2% +sp:0
python3.9 cifar10.py --wbits 32 --abits 32 --gpu 3 --save --arch resnet20 --batch_size 128 --dense --lr 1e-1 --sched step --warmup_epochs 0 --decay_epochs 50 --opt SGD --mixup 0


