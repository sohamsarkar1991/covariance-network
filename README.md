# Covariance Networks (CovNet) 
(see **Demo.ipynb** for demonstration)

A `covariance network (CovNet)` model can be built using the module **CovNetworks** by specifying the parameters. Some useful functions for fitting the CovNet model are in **Important_Functions.py**, **Important_Functions_fMRI.py** and **Other_functions.py**.

The main functions are **CovNet.py** and **CovNet_CV.py**. To use these functions, the data should be stored in _Example.dat_ and _locations.dat_. _Example.dat_ contains the observed fields as an NxD matrix, where N is the number of fields and D is the number of locations (grid points) where the fields are observed. _locations.dat_ contains the (D) grid points as an Dxd matrix.

To compute error of the fit, evaluations of the actual covariance should be provided at M locations (u,v). These should be stored in _True_locations.dat_ and _True_cov.dat_. _True_locations.dat_ should contain an Mx2d matrix of locations (u,v) and _True_cov.dat_ should contain the M values of c(u,v) at those locations. This corresponds to the Monte-Carlo approximation of the error as done in the paper. Other methods for error computation are also available in **Other_functions.py**.

The data can be generated using the codes in the folder **Datagen**. Some hyper-parameters for fitting the model (e.g. initialization, number of epochs etc.) should be set in **current_setup.py**. Check _Datagen/Simulation setup.ods_ for the hyper-parameters used in simulations.

For the analysis of fMRI data set in the paper, we used the functions **CovNet_fMRI.py** and **CovNet_fMRI_CV.py**. The data (after preprocessing) were stored as a nifti image _Data.nii.gz_. The corresponding locations (grid points) were also stored as a nifti image _locations_3D.nii.gz_.

In all the cases, the codes produce the errors as a _.txt_ file and the fitted models as a _pickle_ (_.pt_) file. After running the code, the fitted model can be accessed as **model**, or can be read from the _.pt_ file. This can be used with the function _PCA_ in the **CNet_PCA** module to obtain the eigendecomposition of the fitted CovNet.

The cross-validation (CV) version **CovNet_CV.py** computes the CV-score for some specific choices of the hyperparameters, finds the best hyperparameter choice by minimizing the CV-score, and fits the best model to the full data.

**CovNet_fMRI_CV.py** computes the CV-score for a specific hyperparameter and also fits the same model to the full data. This facilitates parallelization.

## Reference
Sarkar, S. & Panaretos, V. M. (2021). CovNet: Covariance Networks for Functional Data on Multidimensional Domains. arXiv preprint arXiv:2104.05021.
