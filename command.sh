nohup python main.py --exp_id ti1.0_rest13 -p_b 3.5 -router_cp ./workspace/tinyimagenet/ti_ensemble/checkpoint_experts/set1.pth.tar -router_cp_icc ./workspace/tinyimagenet/ti_resnet20_router/model_best.pth.tar -name imagenet -dp ./data/tiny-imagenet-200/   > ti1.0_rest13.log &



nohup python  main.py --exp_id p4.1 --train_batch 32  -p_b 2.0  -router_cp ./workspace/pets/pets_resnet20_icc/model_best.pth.tar -router_cp_icc ./workspace/pets/pets_resnet20_icc/model_best.pth.tar -dp ./data/pets37/ -co 4  -name pets  > p4.1 & 

nohup python  main.py --exp_id p4.0 --train_batch 32  -p_b 3.5  -router_cp ./workspace/pets/pets_resnet20_icc/model_best.pth.tar -router_cp_icc ./workspace/pets/pets_resnet20_icc/model_best.pth.tar -dp ./data/pets37/ -co 4  -name pets  > p4.0 &

python main.py  --exp_id c10_3.0  -p_b 3 -router_cp ./workspace/cifar10/c10_icc_router_resnet20/model_best.pth.tar  -router_cp_icc  ./workspace/cifar10/c10_icc_router_resnet20/model_best.pth.tar  -name cifar10 -dp ./data/cifar10png/  > c10_3.0.log &


nohup python main.py  --exp_id c10_3.0_R  -p_b 3 -co 20  -router_cp ./workspace/cifar10/c10_icc_router_resnet20/model_best.pth.tar  -router_cp_icc  ./workspace/cifar10/c10_icc_router_resnet20/model_best.pth.tar  -name cifar10 -dp ./data/cifar10png/  > c10_3.0_R.log &