rem pretrain, joint, pmnist
python main.py --cuda --pre_epochs 50 --epochs 250 --save joint_pMNIST300_1_pre.pt --dataset pMNIST --seed 1111 --load_pretrained

rem no_pre, joint, mnist
python main.py --cuda --pre_epochs 0 --epochs 250 --save joint_MNIST300_1.pt --dataset MNIST --seed 1111

rem no_pre, joint, pmnist
python main.py --cuda --pre_epochs 0 --epochs 250 --save joint_pMNIST300_1_nopre.pt --dataset pMNIST --seed 1111

rem no_pre, single, pmnist
python main.py --cuda --pre_epochs 0 --epochs 250 --save single_pMNIST300_1_nopre.pt --dataset pMNIST --seed 1111