# MS-NET-v2
Modular architectures enhance Deep Neural Networks (DNNs) by reducing error rates, enabling uncertainty estimation, and increasing inference efficiency through selective module execution. While modular ensemble methods like bagging, boosting, and stacking improve DNNs performance, they are often computationally intensive. Adaptive inference strategies, such as the Modular Selective Network (MS-NET), address these issues by using independent router and expert modules, allowing parallel training and selective inference. In this study, we introduce MS-NET-v2, an optimized version of MS-NET, which enhances the construction of subsets and the expert training methodology. First, we introduce a cut-off variable, O, which systematically limits the sampling of binary class pairs from the Inter-Class Correlation (ICC) matrix. Next, we propose a subset merging algorithm that generates multi-class subsets from these binary subsets. This algorithm creates subsets that encode coarse concepts, in contrast to the fine-grained concepts represented by binary subsets. Based on loss landscape theory and to exploit the non-convexity of DNNs, we train these experts from scratch on multi-class subsets. This approach enhances diversity by covering several unique local minima , resulting in improved collective accuracy. We conduct extensive empirical studies with MS-NET-v2 on the CIFAR-10, CIFAR-100, Tiny-ImageNet, and Pets datasets. To verify the enhanced diversity of the expert networks, we perform function and weight space analyses on MS-NET-v2 experts. These studies demonstrate that MS-NET-v2 significantly improves collective accuracy, expert networks' diversity, and parameter efficiency compared to its predecessor. Additionally, MS-NET-v2 outperforms heavier ensemble methods in both single and multi expert settings.


## Inference phase

1. Clone the repository
```bash
git clone https://github.com/intisar1020/DeepExpertEnsembles
cd DeepExpertEnsembles
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Prepare datasets
Download and extract the datasets (CIFAR-10, CIFAR-100, Tiny-ImageNet, Pets) into the `datasets` directory.
```bash
mkdir datasets
# Download and extract datasets into the datasets directory
# Ensure the datasets are organized as required by the code
```
4. Run the inference script (this just an example, modify parameters as needed)
```bash
python inference.py --exp_id c100_5.1 --dataset_path ./datasets/cifar100png/ --data_name cifar100 --router_cp ./workspace/cifar100/c100_resnet20_router/model_best.pth.tar  --topk 2
```

## Resources
download the checkpoints from [here](https://drive.google.com/drive) and place them in the `workspace` directory. 


## Training phase
TBD
