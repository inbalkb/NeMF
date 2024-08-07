# NeMF: Neural Microphysics Fields


## Abstract
Inverse problems in scientific imaging often seek physical characterization of heterogeneous scene materials. The scene is thus represented by physical quantities, such as the density and sizes of particles (microphysics) across a domain. Moreover, the forward image formation model is physical. An important case is that of clouds, where microphysics in three dimensions (3D) dictate the cloud dynamics, lifetime and albedo, with implication to Earths' energy balance, solar power generation and precipitation. Current methods, however, recover very degenerate representations of microphysics. To enable 3D volumetric recovery of all the required microphysical parameters, we present neural microphysics fields (NeMF). It is based on a deep neural network, whose input is multi-view polarization images. For fast inference, it is pre-trained  through supervised learning. Training relies on polarized radiative transfer, and noise modeling in polarization-sensitive sensors. The results offer unprecedented recovery, including of droplet effective variance. We test NeMF in rigorous simulations and demonstrate it using real-world polarization-image data.

![VIP-CT](readme_files/main_net_figure_train_and_infer2.png)

## Description
This repository contains the official implementation of NeMF: Neural Microphysics Fields, accepted for publication in IEEE Transactions on Pattern Analysis and Machine Intelligence, and presented at ICCP 2024.
Our framework preforms fast scattering tomography of clouds for variable viewing
geometries and resolutions. The core of NeMF is a decoder, which consists of 3 heads. Each of the heads assigns an estimated value of a cloud's microphysical parameters at a queried location: the cloud's effective radius, effective variance and liquid water content. 
The decoder's input is a feature vector. It consists of 3 concatenated feature vectors of 3 encoders: The first is a feature vector expressing the 3D geometry of the queried location. 
The second expresses the geometry of the viewpoint (camera) locations. The third is a vector of image
features, associated with the queried location.  For more details see our [paper](--) and [supplementary material](--).

![results](readme_files/3d_scatter.png)
