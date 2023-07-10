import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size = 3,
                 padding = 1,
                 norm_layer = None,
                 relu = None,
                ):
        '''
        norm_layer:default BatchNorm3d
        relu:default LeakyReLU
        '''
        super().__init__()
        
        if norm_layer == None:
            norm_layer = nn.BatchNorm3d
        if relu == None:
            self.relu = nn.LeakyReLU()
        
        self.skip_connect = None
        '''
        If the number of input channel is not equal to the number of output channel, it's necessary to unify them in the residual connection step.
        So we need to use 1*1 convolution kernel to make the input channel number equal to the output channel number.
        '''
        if in_channel != out_channel:
            self.skip_connect = nn.Conv3d(in_channels=in_channel,
                                          out_channels=out_channel,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
            
        # no reduction in image resolution after convolutional layer
        self.conv1 = nn.Conv3d(in_channels=in_channel,
                               out_channels=out_channel,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=padding)
        
        self.bn1 = norm_layer(out_channel)
        # self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv3d(in_channels=out_channel,
                               out_channels=out_channel,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=padding)
        
        self.bn2 = norm_layer(out_channel)
        
        
        
    def forward(self,x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.skip_connect != None:
            identity = self.skip_connect(identity)
        out = x + identity
        out = self.relu(out)
        return out
    
    
class OurModel(nn.Module):
    '''
    
    
    使用resnet-like网络提取图像特征，经过全局平均池化后拼接domain knowledge；
    在训练时将不同站点的图片特征送入不同的分类器（MLP）中；
    在测试是将测试站点的图片特征送入全部分类器，将结果平均后输出
    '''
    def __init__(self,
                block_num_list = None,
                channel_list = None,
                num_class = 2,
                domain_knowledge = False,
                domain_knowledge_len = None,
                conv_pooling = True,
                train_sites_num = None,
                mlp_ratio = 1.):
        '''
        block_num_list: number of basic block in each layer
        channel_list: number of feature map channel in each layer
        num_class: number of classification
        domain_knowledge: availability of mutil-model domain knowledge (default None, temporarily not used)
        domain_knowledge_len: the length of mutil-model domain knowledge(default None, temporarily not used)
        norm_layer: default batchNorm
        relu: default LeakeyReLU
        mlp_ratio: ratio of the number of neurons in the second layer to the first layer in MLP block
        '''
        super().__init__()

        self.domain_knowledge = domain_knowledge
        self.train_sites_num = train_sites_num
        
        self.conv_pooling = conv_pooling
        
        self.conv = nn.Conv3d(in_channels=channel_list[0],
                             out_channels=channel_list[1],
                             kernel_size=7,
                             stride=2,
                             padding=3)
        
        self.pool = nn.MaxPool3d(kernel_size=2,stride=2,padding=0)
        
        self.layer1 = self._make_layer(in_channel=channel_list[1],
                                      out_channel=channel_list[1],
                                      block_num=block_num_list[0])
        
        self.pool1 = nn.Conv3d(in_channels=channel_list[1],
                              out_channels=channel_list[1],
                              kernel_size=2,
                              stride=2,
                              padding=0)
        
        self.layer2 = self._make_layer(in_channel=channel_list[1],
                                      out_channel=channel_list[2],
                                      block_num=block_num_list[1])
        
        self.pool2 = nn.Conv3d(in_channels=channel_list[2],
                              out_channels=channel_list[2],
                              kernel_size=2,
                              stride=2,
                              padding=0)
        
        self.layer3 = self._make_layer(in_channel=channel_list[2],
                                      out_channel=channel_list[3],
                                      block_num=block_num_list[2])
        
        self.pool3 = nn.Conv3d(in_channels=channel_list[3],
                              out_channels=channel_list[3],
                              kernel_size=2,
                              stride=2,
                              padding=0)
        
        self.layer4 = self._make_layer(in_channel=channel_list[3],
                                      out_channel=channel_list[4],
                                      block_num=block_num_list[3])
        
        self.gav = nn.AdaptiveAvgPool3d(1)
        
        self.mutil_MLP = []
        
 
        if domain_knowledge == True:
 
            for i in range(len(train_sites_num)):
                mlp_temp = nn.Sequential(
                    nn.Linear(in_features=channel_list[4] + domain_knowledge_len,
                             out_features=int((channel_list[4] + domain_knowledge_len) * mlp_ratio)),
                    
                    nn.ReLU(),
                    
                    nn.Linear(in_features=int((channel_list[4] + domain_knowledge_len) * mlp_ratio),
                             out_features=num_class)
                )
                
                self.mutil_MLP.append(mlp_temp.cuda())
        
        else:
            for i in range(len(train_sites_num)):
                mlp_temp = nn.Sequential(
                    nn.Linear(in_features=channel_list[4],
                             out_features=int(mlp_ratio * channel_list[4])),
                    
                    nn.ReLU(),
                    
                    nn.Linear(in_features=int(mlp_ratio * channel_list[4]),
                             out_features=num_class)
                )
                
                self.mutil_MLP.append(mlp_temp.cuda())
            
        
        
    def forward(self,
                x,
                train = True,
                site = None,
                gender = None,
                age = None,
                MMSE = None
               ):
        
        '''
        x: MRI image
        train: if true, feed features to different classifiers base on the site information of MRI Images;
               if False, feed features to all classifiers and average the output of all classifiers as the final output
        site: site information of MRI image
        gender: patients'gender (default None, temporarily not used)
        age: patients'age (default None, temporarily not used)
        MMSE: patients'MMSE score (default None, temporarily not used)
        '''
        
        if self.conv_pooling==False:
            # Do not use convolution instead of maxpooling
            x = self.conv(x)
            x = self.layer1(x)
            x = self.pool(x)
            x = self.layer2(x)
            x = self.pool(x)
            x = self.layer3(x)
            x = self.pool(x)
            x = self.layer4(x)
            x = self.gav(x)
            x = x.squeeze()


        else:
            # Use convolution instead of maxpooling
            x = self.conv(x)
            x = self.layer1(x)
            # print(x.shape)
            x = self.pool(x)
            # print(x.shape)
            x = self.layer2(x)
            # print(x.shape)
            x = self.pool(x)
            # print(x.shape)
            x = self.layer3(x)
            # print(x.shape)
            x = self.pool(x)
            # print(x.shape)
            x = self.layer4(x)
            # print(x.shape)
            x = self.gav(x)
            # print("before squeeze:",x.shape)
            x = x.squeeze()
            if len(x.shape) <= 3:
                x = x.unsqueeze(dim = 0)
          
        
        if self.domain_knowledge == True:
            if site != None:
                site_onehot = F.one_hot(site,num_classes=7)
                x = torch.cat([x,site_onehot],dim = -1)
            if gender != None:
                gender_onehot = F.one_hot(gender,num_classes=2)
                x = torch.cat([x,gender_onehot],dim = -1)
            if age != None:
                x = torch.cat([x,age],dim = -1)
            if MMSE != None:
                x = torch.cat([x,MMSE],dim = -1)
                
            site = site.cpu().int().numpy()
        if train == True:
            result = None
            # print("training")
            for i in range(len(site)):
                # find classifier index
                mlp_loc = self.train_sites_num.index(site[i])

                
                # print("xi shape:",x[i].shape)
                class_probility = self.mutil_MLP[mlp_loc](x[i]).unsqueeze(0)
                # print(class_probility.shape)
                if result == None:
                    result = class_probility
                else:
                    result = torch.cat((result,class_probility),dim = 0)
            
            return result
                    
      
        else:

            # print("evaluate")
            result = None
            for i in range(len(x)):
                class_probility = None
                for mlp in self.mutil_MLP:
                    if class_probility == None:
                        class_probility = mlp(x[i])
                    else:
                        class_probility += mlp(x[i])
                class_probility /= len(self.mutil_MLP)
                class_probility = class_probility.unsqueeze(0)
                # print(class_probility.shape)
                
                if result == None:
                    result = class_probility
                else:
                    result = torch.cat((result,class_probility),dim = 0)
                    
            return result
        
        
        
        
    def _make_layer(self,
                    in_channel,
                    out_channel,
                    block_num):

        layer = []
        first_block = BasicBlock(in_channel=in_channel,
                                 out_channel=out_channel)
        layer.append(first_block)
        
        for i in range(1,block_num):
            layer.append(BasicBlock(in_channel=out_channel,
                                    out_channel=out_channel))
        return nn.Sequential(*layer)
    