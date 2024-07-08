import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import *
from core.extractor import MultiBasicEncoder, Feature
from core.srstereov1_EENet import *
from core.geometry import *
from core.submodule import *
import time
import pdb
from torchvision import transforms
from torchvision.transforms import Resize
import random

try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class hourglass(nn.Module):
    def __init__(self, in_channels, out_channels=8):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, out_channels, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))

        self.feature_att_8 = FeatureAtt(in_channels*2, 64)
        self.feature_att_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_32 = FeatureAtt(in_channels*6, 160)
        self.feature_att_up_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_up_8 = FeatureAtt(in_channels*2, 64)

    def forward(self, x, features):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)

        return conv

class SRStereov1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        context_dims = args.hidden_dims

        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn="batch", downsample=args.n_downsample)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])

        self.feature = Feature()

        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
            )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24), nn.ReLU()
            )

        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)

        self.conv = BasicConv_IN(96, 96, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)

        self.corr_stem = BasicConv(8, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        self.corr_feature_att = FeatureAtt(8, 96)
        self.cost_agg = hourglass(8)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)

        # SR-Stereo_v1
        self.unnet = UnNet1()

        if args.edge_estimator:
            self.edge_estimator =  EENet1()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def upsample_disp(self, disp, mask_feat_4, stem_2x):

        with autocast(enabled=self.args.mixed_precision):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x) #两种特征concat，torch.Size([4, 32, 80, 184]), torch.Size([4, 32, 160, 368]) -> torch.Size([4, 64, 160, 368])
            spx_pred = self.spx_gru(xspx) #反卷积，torch.Size([4, 9, 320, 736])
            spx_pred = F.softmax(spx_pred, 1)
            up_disp = context_upsample(disp*4., spx_pred).unsqueeze(1)

        return up_disp


    def forward(self, image1, image2, iters=12, disp_gt=None, flow_init=None, test_mode=False):
        """ Estimate disparity between pair of frames """

        image1 = (2 * (image1 / 255.0) - 1.0).contiguous() #torch.Size([4, 3, 320, 736])
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
        with autocast(enabled=self.args.mixed_precision):
            feature_2x_left, features_left = self.feature(image1) #len:4 [torch.Size([4, 48, 80, 184]), torch.Size([4, 64, 40, 92]), torch.Size([4, 192, 20, 46]), torch.Size([4, 160, 10, 23])]
            feature_2x_right, features_right = self.feature(image2)
            stem_2x = self.stem_2(image1) #torch.Size([4, 32, 160, 368])
            stem_4x = self.stem_4(stem_2x) #torch.Size([4, 48, 80, 184])
            stem_2y = self.stem_2(image2)
            stem_4y = self.stem_4(stem_2y)

            features_left[0] = torch.cat((features_left[0], stem_4x), 1) #torch.Size([4, 96, 80, 184])
            features_right[0] = torch.cat((features_right[0], stem_4y), 1) 

            match_left = self.desc(self.conv(features_left[0])) #torch.Size([4, 96, 80, 184])
            match_right = self.desc(self.conv(features_right[0]))
            
            gwc_volume = build_gwc_volume(match_left, match_right, self.args.max_disp//4, 8) # torch.Size([4, 8, self.args.max_disp//4, 80, 184]), self.args.max_disp//4 = 48
            gwc_volume = self.corr_stem(gwc_volume) # k=3的3d卷积， torch.Size([4, 8, 48, 80, 184]), 3d卷积可能会破坏映射关系？不会，平移是在DHW三个维度进行的，可以考虑使用1*3*3的卷积核。不过3*3*3的卷积核可能存在相对关系的隐式建模。
            gwc_volume = self.corr_feature_att(gwc_volume, features_left[0]) # 注意力, torch.Size([4, 8, 48, 80, 184])
            geo_encoding_volume = self.cost_agg(gwc_volume, features_left) # 增加注意力的3D沙漏型网络, torch.Size([4, 8, 48, 80, 184])

            # Init disp from geometry encoding volume
            prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1) # 视差概率图, torch.Size([4, 8, 48, 80, 184]) -> torch.Size([4, 1, 48, 80, 184]) -> torch.Size([4, 48, 80, 184])
            init_disp = disparity_regression(prob, self.args.max_disp//4) # 根据视差概率图加权求和, torch.Size([4, 1, 80, 184])
            
            del prob, gwc_volume

            if not test_mode:
                xspx = self.spx_4(features_left[0]) #torch.Size([4, 24, 80, 184])
                xspx = self.spx_2(xspx, stem_2x) #将xspx先卷积，再上采样，最后与stem_2x concat在一起，torch.Size([4, 64, 160, 368])
                spx_pred = self.spx(xspx) #反卷积，torch.Size([4, 9, 320, 736])
                spx_pred = F.softmax(spx_pred, 1) #torch.Size([4, 9, 320, 736])

            cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers) # res block特征提取, 输送至GRU, [[torch.Size([4, 128, 80, 184])]*2, [torch.Size([4, 128, 40, 92])]*2, [torch.Size([4, 128, 20, 46])]*2 ]
            net_list = [torch.tanh(x[0]) for x in cnet_list] #范围[-1, 1], [torch.Size([4, 128, 80, 184]), torch.Size([4, 128, 40, 92]), torch.Size([4, 128, 20, 46])]
            inp_list = [torch.relu(x[1]) for x in cnet_list] #范围[0, 1], [torch.Size([4, 128, 80, 184]), torch.Size([4, 128, 40, 92]), torch.Size([4, 128, 20, 46])]
            inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)] #[[torch.Size([4, 128, 80, 184])]*3, [torch.Size([4, 128, 40, 92])]*3, [torch.Size([4, 128, 20, 46])]*3 ]

        geo_block = Combined_Geo_Encoding_Volume
        geo_fn = geo_block(self.args.local_pairs_correlation, match_left.float(), match_right.float(), geo_encoding_volume.float(), radius=self.args.corr_radius, num_levels=self.args.corr_levels)
        
        b, c, h, w = match_left.shape
        coords = torch.arange(w).float().to(match_left.device).reshape(1,1,w,1).repeat(b, h, 1, 1)
        disp = init_disp # torch.Size([4, 1, 80, 184])
        disp_preds = []

        if not test_mode:
            delta_disps = []

        # GRUs iterations to update disparity
        for itr in range(iters): # train: 22, valid: 32
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords) # CGEV特征, torch.Size([4, 162, 80, 184])

            uc_map_now, _ = self.unnet(geo_feat)  # torch.Size([4, 1, 80, 184])

            with autocast(enabled=self.args.mixed_precision):
                # 来自左图，net_list：范围[-1, 1], [torch.Size([4, 128, 80, 184]), torch.Size([4, 128, 40, 92]), torch.Size([4, 128, 20, 46])]
                # 来自左图，inp_list：[[torch.Size([4, 128, 80, 184])]*3, [torch.Size([4, 128, 40, 92])]*3, [torch.Size([4, 128, 20, 46])]*3 ]
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru: # Update low-res ConvGRU
                    net_list = self.update_block(net_list, inp_list, iter16=True, iter08=False, iter04=False, update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru: # Update low-res ConvGRU and mid-res ConvGRU
                    net_list = self.update_block(net_list, inp_list, iter16=self.args.n_gru_layers==3, iter08=True, iter04=False, update=False)

                net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, iter16=self.args.n_gru_layers==3, iter08=self.args.n_gru_layers>=2) 

                # SR-Stereo_v1
                delta_disp = self.args.xga_uncertain_aft_m * torch.tanh(delta_disp/self.args.xga_uncertain_aft_m) * (1.0 + 0.5*uc_map_now)

            if not test_mode and self.args.stepwise:  
                delta_disps.append([delta_disp, disp])

            disp = disp + delta_disp
            if test_mode and itr < iters-1:
                continue

            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)
            disp_preds.append(disp_up)

        if test_mode:
            return disp_up

        edge_map = None 
        if self.args.edge_estimator:

            # 上采样加权, init_disp：torch.Size([4, 1, 80, 184])， spx_pred:[0,1] torch.Size([4, 9, 320, 736]) -> torch.Size([4, 80, 184]) -> torch.Size([4, 1, 80, 184])
            init_disp = context_upsample(init_disp*4., spx_pred.float()).unsqueeze(1)
            if self.args.xga_prob:
                init_disp_filter = context_upsample(init_disp_filter*4., spx_pred.float()).unsqueeze(1)
                
            edge_map, _ = self.edge_estimator(init_disp, image1) # self.edge_estimator(image1), init_disp， disp_up    

        # 上采样加权, init_disp：torch.Size([4, 1, 80, 184])， spx_pred:[0,1] torch.Size([4, 9, 320, 736]) -> torch.Size([4, 80, 184]) -> torch.Size([4, 1, 80, 184])
        init_disp = context_upsample(init_disp*4., spx_pred.float()).unsqueeze(1)

        # init_disp: torch.Size([4, 1, 320, 736])    disp_preds: [torch.Size([4, 1, 320, 736])] * iters
        return init_disp, disp_preds, edge_map, delta_disps
