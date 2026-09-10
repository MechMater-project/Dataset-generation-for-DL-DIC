This repository provides codes to generate datasets for DL-DIC as well as Ncorr DIC code to calculate DIC automatically for large sets of data. This repository is linked to the work "Effects of Dataset Design on Deep Learning Digital Image Correlation Performance". Please  cite the repository as well as the aforementioned reference.

For the three supervised CNN DL-DIC, see their original repository for model download and training.
R3Net: https://github.com/LianpoWang/R3-DICnet/blob/main/train.py
StrainNet-f: https://github.com/DreamIP/StrainNet/tree/master
DICTr: https://github.com/vincentjzy/dictr

Supervised DL-DIC neural networks, requires for training a dataset with reference images, deformed images and displacement fields in horizontal direction (U) and vertical direction (V)
Here we propose, six different  [[Dataset generation method]] as well as a combination of two methods for the seventh.

Models's performance is assess through the [[Loss function]].

We used open source Ncorr DIC for comparison [[DL-DIC vs. Classic DIC]]




