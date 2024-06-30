# SR-Stereo
python train_stereo.py --logdir ./checkpoints/sceneflow_srstereo --train_datasets sceneflow --num_steps 50000 --batch_size 4 --train_iters 10 --valid_iters 15 --lr 0.0001 --stepwise --xga_uncertain --xga_uncertain_type 23 --xga_self_itr_uctest --xga_uncertain_aft --xga_uncertain_aft_type 6 --xga_uncertain_aft_m 2 --disp_gt_weight --disp_gt_weight_type 9 --disp_gt_weight_h1 0.5

# SR-Stereo + edge_estimator
python train_stereo.py --logdir ./checkpoints/sceneflow_srstereo_edge_estimator --train_datasets sceneflow --num_steps 50000 --batch_size 4 --train_iters 10 --valid_iters 15 --lr 0.0001 --stepwise --xga_uncertain --xga_uncertain_type 23 --xga_self_itr_uctest --xga_uncertain_aft --xga_uncertain_aft_type 6 --xga_uncertain_aft_m 2 --disp_gt_weight --disp_gt_weight_type 9 --disp_gt_weight_h1 0.5 --edge_estimator

# DAPE
python train_stereo.py --logdir ./checkpoints/kitti_srstereo_edge_supervised --train_datasets kitti --restore_ckpt ./checkpoints/sceneflow_srstereo/sr-stereo.pth --num_steps 50000 --batch_size 4 --train_iters 10 --valid_iters 15 --lr 0.0001 --stepwise --xga_uncertain --xga_uncertain_type 23 --xga_self_itr_uctest --xga_uncertain_aft --xga_uncertain_aft_type 6 --xga_uncertain_aft_m 2 --disp_gt_weight --disp_gt_weight_type 9 --disp_gt_weight_h1 0.5 --edge_supervised
