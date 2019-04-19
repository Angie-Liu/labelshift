## Label Shift 
### This tutorial walks you through the implementation of Regularized Learning under Label Shifts (RLLS).

The model constructed in this tutorial follows the work described in [Regularized Learning for Domain Adaptation under Label Shifts](https://arxiv.org/pdf/1903.09734.pdf). This work proposes RLLS, a novel algorithm for domain adaptation in the presence of a shift in the label distribution. This work is among those few works which theoretically analyze this problem and provides a good generalization guarantee for the RLLS without prior knowledge, required by earlier works. The RLLS is designed in the insight of the theoretical development in this paper.


  * Requirements: [cvxpy](https://www.cvxpy.org/install/), [gurobi](http://www.gurobi.com)
  * Comment: 
       Code-base for training: offline_label_shift.py online_label_shift

       Creat artificial shifts: mnist_for_labelshift.py cifar10_for_labelshift.py --for generating shifts in data

       Code-base for testing: label_shift.py, w_comp.py --

