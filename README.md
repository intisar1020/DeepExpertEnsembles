# MS-NET-v2
## Modular architectures enhance Deep Neural
Networks (DNNs) by reducing error rates, enabling un-
certainty estimation, and increasing inference efficiency
through selective module execution. While modular en-
semble methods like bagging, boosting, and stacking
improve DNNs performance, they are often computa-
tionally intensive. Adaptive inference strategies, such
as the Modular Selective Network (MS-NET), address
these issues by using independent router and expert
modules, allowing parallel training and selective infer-
ence. In this study, we introduce MS-NET-v2, an op-
timized version of MS-NET, which enhances the con-
struction of subsets and the expert training method-
ology. First, we introduce a cut-off variable, O, which
systematically limits the sampling of binary class pairs
from the Inter-Class Correlation (ICC) matrix. Next,
we propose a subset merging algorithm that generates
multi-class subsets from these binary subsets. This al-
gorithm creates subsets that encode coarse concepts, in
contrast to the fine-grained concepts represented by bi-
nary subsets. Based on loss landscape theory and to ex-
ploit the non-convexity of DNNs, we train these experts
from scratch on multi-class subsets. This approach en-
hances diversity by covering several unique local min-
ima, resulting in improved collective accuracy. We con-
duct extensive empirical studies with MS-NET-v2 on
the CIFAR-10, CIFAR-100, Tiny-ImageNet, and Pets
datasets. To verify the enhanced diversity of the expert
networks, we perform function and weight space analy-
ses on MS-NET-v2 experts. These studies demonstrate
that MS-NET-v2 significantly improves collective ac-
curacy, expert diversity, and parameter efficiency com-
pared to its predecessor. Additionally, MS-NET-v2 out-
performs heavier ensemble methods in both single and
multi-expert settings.
