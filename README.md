# SR-Stereo & DAPE

<p align="center">
  <h1 align="center">Stepwise Regression and Pre-trained Edges for Practical Stereo Matching</h1>
  <h3 align="center"><a href="https://ieeexplore.ieee.org/abstract/document/11089999">T-ITS 2025</a>
  <div align="center"></div>
</p>

> [Weiqing Xiao](https://github.com/zhuxing0), Wei Zhao* <br> Behang University

The stepwise regression architecture: 
-----
The iteration-based methods regress disparity error by predicting residual disparity ∆dk, while SR-Stereo splits the disparity error into
multiple segments and regresses them by predicting multiple disparity clips.
<p align="center">
  <a href="">
    <img src="https://github.com/zhuxing0/SR-Stereov1-DAPE/blob/main/img_from_paper/idea.png" alt="Logo" width="70%">
  </a>
</p>

The proposed SR-Stereo:
-----
Compared to iteration-based methods, SR-Stereo is specially designed in terms of the update unit and the regression objective. Specifically, we propose a stepwise regression unit that outputs range-controlled disparity clips, rather than unconstrained residual disparities. Further, we design separate regression objectives for each stepwise regression unit, instead of simply using the disparity error.
<p align="center">
  <a href="">
    <img src="https://github.com/zhuxing0/SR-Stereov1-DAPE/blob/main/img_from_paper/SR-Stereo.png" alt="Logo" width="80%">
  </a>
</p>

The overall framework of the proposed DAPE:
-----
First, a robust stereo model SR-Stereo and a lightweight edge estimator are pre-trained on a large synthetic dataset with dense ground truth. Then, we use the pre-trained SR-Stereo and edge estimator to generate the edge map of target domain, where the background pixels (i.e., non-edge region pixels) are used as edge pseudo-labels. Finally, we jointly fine-tune the pre-trained SR-Stereo using the edge pseudo-labels and sparse ground truth disparity.
<p align="center">
  <a href="">
    <img src="https://github.com/zhuxing0/SR-Stereov1-DAPE/blob/main/img_from_paper/DAPE.png" alt="Logo" width="80%">
  </a>
</p>



# vis
Domain-adaptive visualization on KITTI:
![image](https://github.com/zhuxing0/SR-Stereov1-DAPE/blob/main/img_from_paper/vis.png)

Qualitative disparity estimation results of DAPE on ETH3D:
![image](https://github.com/zhuxing0/SR-Stereov1-DAPE/blob/main/img_from_paper/DAPE_vis1.png)

Qualitative disparity estimation results of DAPE on KITTI test set:
![image](https://github.com/zhuxing0/SR-Stereov1-DAPE/blob/main/img_from_paper/DAPE_vis2.png)
