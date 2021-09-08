from functools import partial
from .losses import *
import re
from collections import OrderedDict

import torch
import torch.nn as nn 
import torch.nn.functional as F 

# class Extractor(nn.Module):
#     def __init__(self, student, teacher, layers):
#         super().__init__()

#         self.teacher_features = []
#         self.student_features = []
#         self.channel_dims = []  # student 和 teacher 被提取层的输出通道数
#         self.total_dims = []  # student 和 teacher 被提取层的输出维数

#         for i,(student_layer,teacher_layer,channel_dim,total_dim) in enumerate(layers):
#             self.channel_dims.append(channel_dim)
#             self.total_dims.append(total_dim)

#             if not isinstance(teacher_layer,list):
#                 teacher_layer = [teacher_layer]

#             for name, module in teacher.named_modules():
#                 if name in teacher_layer:
#                     module.register_forward_hook(partial(self.hook_fn_forward, name=name, type='teacher',layer_num=i))
#                     print(f'teacher_layer :{teacher_layer} hooked!!!!')
#             for name, module in student.named_modules():
#                 if name == student_layer:
#                     module.register_forward_hook(partial(self.hook_fn_forward, name=name, type='student'))
#                     print(f'student_layer :{student_layer} hooked!!!!')

#     def hook_fn_forward(self, module, input, output, name, type,layer_num=None):
#         if self.training == True:
#             if 'norm' in name or 'fc' in name:
#                 output = output.permute(0,2,1)

#             if type == 'student':
#                 self.student_features.append(output)
#             if type == 'teacher':
#                 # if len(self.teacher_features)>layer_num:
#                 #     self.teacher_features[layer_num].append(output)
#                 # else:
#                 #     self.teacher_features.append([output])
#                 self.teacher_features.append([output])

class Extractor(nn.Module):
    def __init__(self,student,teacher):
        super().__init__()

        self.teacher_features = [[] for i in range(4)]
        self.student_features = [[] for i in range(4)]

        self.teacher_attns = [[] for i in range(4)]
        self.student_attns = [[] for i in range(4)]

        for name, module in teacher.named_modules():
            if 'backbone' not in name:
                continue
            if 'mlp.fc2' in name:
                type = 'feature'
            elif 'attn.attn_drop' in name:
                type = 'attn'
            else:
                continue

            if 'block1' in name:
                block = 0
            elif 'block2' in name:
                block = 1
            elif 'block3' in name:
                block = 2
            elif 'block4' in name:
                block = 3
            else:
                continue
            module.register_forward_hook(
                partial(self.hook_fn_forward,
                 name=name, 
                 type=type,
                 block=block,
                 model = 'teacher'))

        for name, module in student.named_modules():
            if 'backbone' not in name:
                continue
            if 'mlp.fc2' in name:
                type = 'feature'
            elif 'attn.ATTN' in name:
                type = 'attn'
            else:
                continue

            if 'block1' in name:
                block = 0
            elif 'block2' in name:
                block = 1
            elif 'block3' in name:
                block = 2
            elif 'block4' in name:
                block = 3
            else:
                continue

            module.register_forward_hook(
                partial(self.hook_fn_forward,
                 name=name, 
                 type=type,
                 block=block,
                 model = 'student'))
            
    def hook_fn_forward(self, module, input, output, name, type, block, model):
        if self.training == True:
            if type == 'attn':
                output = output.permute(0,1,3,2)
                B,num_head,C,WH = output.shape
                output = output.reshape(B,C*num_head,WH)
            if type == 'feature':
                output = output.permute(0,2,1)

            if model == 'teacher':
                if type == 'attn':
                    self.teacher_attns[block].append(output)
                if type == 'feature':
                    self.teacher_features[block].append(output)
            if model == 'student':
                if type == 'attn':
                    self.student_attns[block].append(output)
                if type == 'feature':
                    self.student_features[block].append(output)



class ff(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()

        # self.ff = nn.Sequential(
        #     nn.Conv1d(input_size,input_size, kernel_size=1, stride=1, padding=0),
        #     nn.GELU(),
        #     nn.Conv1d(input_size,output_size, kernel_size=1, stride=1, padding=0),
        #     nn.GELU(),
        #     nn.Conv1d(output_size,output_size, kernel_size=1, stride=1, padding=0),
        # )
        
        self.ff = nn.Sequential(
            nn.Linear(input_size,input_size),
            nn.GELU(),
            nn.Linear(input_size,output_size),
            nn.GELU(),
            nn.Linear(output_size,output_size),
        )
    def forward(self,x):
        x = self.ff(x)
        return x

class conv1d(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()
        self.conv = nn.Conv1d(input_size,output_size, kernel_size=1, stride=1, padding=0)
    def forward(self,x):
        x = x.permute(0,2,1)
        x = self.conv(x)
        x = x.permute(0,2,1)
        return x

class SR(nn.Module):
    def __init__(self,WHn,C):
        super().__init__()
        self.conv = nn.Conv1d(WHn*C//256,WHn//16,1)
    def forward(self,x):
        B, n_t, num_head, WH, C = x.shape
        x = x.permute(0,1,3,2,4).reshape(B*n_t,WH,C*num_head)
        x = x.reshape(B*n_t,256,-1).permute(0,2,1)
        x = self.conv(x).permute(0,2,1)
        x = x.reshape(B,n_t,-1) # [B,n_t,256]
        return x

# class Attention(nn.Module):
#     def __init__(self,WHn,C_s,C_t):
#         super().__init__()
#         self.sr_q = SR(WHn,256)
#         self.sr_k = SR(WHn,256)

#         self.WHn = WHn

#     def forward(self,q,k,v):
#         # q : [b,num_head,WH,C_s]
#         # k : [b,n_t,num_head,WH,C]
#         # v : [b,n_t,WH,C]

#         B, num_head, WH, C = q.shape
#         n_t = v.shape[1]

#         q = q.unsqueeze(1)
#         q = self.sr_q(q) # [B,1,256]

#         k = self.sr_k(k) # [B,n_t,256]

#         attn = q @ (k.permute(0,2,1))
#         attn = attn.softmax(dim=2) # # [B,1,n_t]

#         v = v.reshape(B,n_t,-1)
#         x = (attn @ v).squeeze(-1)# [B,WH*C]

#         return x,attn.squeeze(1).mean(dim=0)

class Attention(nn.Module):
    def __init__(self,attn_shape,feature_shape,R):
        super().__init__()
        B,N_attn,C_attn = attn_shape
        B,N_feature,C_feature = feature_shape

        self.R = R
        # self.Q = nn.Conv1d(C_attn,C_attn//16,kernel_size=1)
        # self.K = nn.Conv1d(C_attn,C_attn//16,kernel_size=1)
        # self.V = nn.Conv1d(C_feature,C_feature,kernel_size=1)

        self.Q = nn.Linear(C_attn*R,C_attn)
        self.K = nn.Linear(C_attn*R,C_attn)
        self.V = nn.Linear(C_feature,C_feature)

        self.scale = C_attn**-0.5

        self.attn = None
    def forward(self,q,k,v):
        # q:[b,C_attn,N_attn]
        # k:[b,num_t,C_attn,N_attn]
        # v:[b,num_t,C_feature,N_feature]
        b,num_t,C_attn,N_attn = k.shape
        b,num_t,C_feature,N_feature = v.shape

        # print(C_attn,N_attn,self.R)
        # print(C_attn,N_attn/self.R)

        q = q.permute(0,2,1).detach()
        q = q.reshape(b,N_attn//self.R,C_attn*self.R) 
        q = self.Q(q)# q:[b,N_attn/R,C_attn]

        k = k.permute(0,1,3,2).detach()
        k = k.reshape(b,num_t,N_attn//self.R,C_attn*self.R)
        k = self.K(k)# k:[b,num_t,N_attn/R,C_attn]

        q = q.reshape(b,1,-1)
        k = k.reshape(b,num_t,-1)
        v = v.reshape(b,num_t,-1)
        attn = q @ (k.permute(0,2,1)) * self.scale  # [b,1,num_t]

        # q = q.unsqueeze(2)
        # k = k.permute(0,2,1,3)
        # v = v.reshape(b,num_t,-1)
        # attn = q @ (k.permute(0,1,3,2)) * self.scale
        # attn = attn.mean(dim=1)
        
        
        attn = (attn).softmax(dim=2)
        attn = (0.99*self.attn + 0.01*attn) if (self.attn is not None) else attn
        # self.attn = attn
        x = (attn @ v).reshape(b,C_feature,N_feature)
        self.attn = attn.detach()
        return x,self.attn.mean(dim=0).squeeze()

class AttnAdaptor(nn.Module):
    def __init__(self, attn_shape,feature_shape,R):
        super().__init__()
        self.conv = nn.Conv1d(feature_shape[2],feature_shape[2],kernel_size=1)
        self.attention = Attention(attn_shape,feature_shape,R)

        self.norm = nn.LayerNorm(feature_shape[1:][::-1],elementwise_affine=True)
        # self.norm = nn.InstanceNorm1d(feature_shape[-1])
    #     self.apply(self._init_weights)
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         from timm.models.layers import  trunc_normal_
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()
    def forward(self, student_feature,student_attn,teacher_features,teacher_attns):
        teacher_features = [i.unsqueeze(1) for i in teacher_features]
        teacher_attns = [i.unsqueeze(1) for i in teacher_attns]

        teacher_feature = torch.cat(teacher_features,dim = 1) # [b,n_t,C,WH]
        teacher_attn = torch.cat(teacher_attns,dim = 1)# [b,n_t,C,num_head*WH]
        teacher_feature, attn = self.attention(student_attn,teacher_attn,teacher_feature)

        student_feature = self.conv(student_feature)

        student_feature = self.norm(student_feature)
        teacher_feature = self.norm(teacher_feature)

        return student_feature,teacher_feature,attn

class Adaptor(nn.Module):
    def __init__(self,input_size,output_size,total_dim):
        super().__init__()
        self.total_dim = total_dim

        ff = nn.Conv1d(input_size,input_size, kernel_size=1, stride=1, padding=0)

        if total_dim == 3:
            # self.ff = nn.Conv1d(input_size,output_size, kernel_size=1, stride=1, padding=0)
            self.ff = ff
        elif total_dim == 4:
            self.ff = nn.Conv2d(input_size,output_size, kernel_size=1, stride=1, padding=0)

    def forward(self,x,x_teacher,gt_semantic_seg):
        if self.total_dim == 2:
            x = x
        elif self.total_dim == 3:
            # x = x.permute(0,2,1)
            x = self.ff(x)
            # x = x.permute(0,2,1)
        elif self.total_dim == 4:
            x = self.ff(x)
        else:
            raise ValueError('wrong total_dim')
        return x,x_teacher

class LogitsAdaptor(nn.Module):
    def __init__(self,input_size,output_size,distill_strategy):
        super().__init__()
        self.distill_strategy = distill_strategy
        self.ff = self.ff = nn.Conv2d(input_size,output_size, kernel_size=1, stride=1, padding=0)
    def forward(self,x_student,x_teacher,x_gt):
        from mmseg.ops import resize
        # print(x_student.shape)
        # print(x_teacher.shape)
        # print(x_gt.shape)
        x_student = resize(
            input=x_student,
            size=x_gt.shape[2:],
            mode='bilinear',
            align_corners=False)
        x_student = self.ff(x_student)
        x_teacher = resize(
            input=x_teacher,
            size=x_gt.shape[2:],
            mode='bilinear',
            align_corners=False)  # [b,C,W,H]  

        if self.distill_strategy == 'distill':
            return x_student,x_teacher
        else:
            b, C, W, H = x_teacher.shape
            x_teacher = x_teacher.permute(1,0,2,3).reshape(C,-1) # [C,b*W*H]
            x_student = x_student.permute(1,0,2,3).reshape(C,-1) # [C,b*W*H]

            label = x_gt.reshape(-1).unsqueeze(0) # # [1,b*W*H]
            mask =( label!=255)
            label = torch.masked_select(label,mask).unsqueeze(0)
            x_teacher = torch.masked_select(x_teacher,mask).reshape(C,-1)
            x_student = torch.masked_select(x_student,mask).reshape(C,-1)

            student_pd = torch.argmax(x_student,dim=0)
            teacher_pd = torch.argmax(x_teacher,dim=0)

            pre1 = (teacher_pd == label).bool()
            pre2 = (student_pd != label).bool()

            if '0' in self.distill_strategy:
                learn_mask = torch.ones(pre1.shape).bool().cuda()
            elif '1' in self.distill_strategy:
                learn_mask = (pre1).bool()
            elif '2' in self.distill_strategy:
                learn_mask = (pre1 & pre2).bool()
            elif 'zero' in self.distill_strategy:
                learn_mask = torch.zeros(pre1.shape).bool().cuda()

            learn_proportion = torch.sum(learn_mask)/(learn_mask.shape[1]*learn_mask.shape[0])

            if learn_proportion.item() == 0:
                return x_student.reshape(C,-1),x_teacher.reshape(C,-1)
            else:
                x_teacher = torch.masked_select(x_teacher,learn_mask)
                x_student = torch.masked_select(x_student,learn_mask)
                return x_student.reshape(C,-1),x_teacher.reshape(C,-1)
class DistillationLoss_(nn.Module):
    def __init__(self,distillation,tau):
        super().__init__()
        R = [64,16,4,1]
        self.kd_loss = CriterionChannelAwareLoss(1)
        student_layers, teacher_layers,attn_shape,feature_shape = distillation['layers']

        self.adaptors = nn.ModuleList()

        for i in range(len(student_layers)):
            self.adaptors.append(nn.ModuleList())
            for _ in range(student_layers[i]):
                self.adaptors[i].append(AttnAdaptor(attn_shape[i],feature_shape[i],R[i]))
    def forward(self,teacher_features, student_features,teacher_attns,student_attns,loss_dict,gt_semantic_seg):
        for i in range(len(student_features)):
            for j in range(len(student_features[i])):
                adaptor = self.adaptors[i][j]
                x_student,x_teacher,attn = adaptor(
                                student_features[i][j],student_attns[i][j],
                                teacher_features[i],teacher_attns[i])

                for k in range(attn.shape[0]):
                    loss_dict.update({f'attn block{i}:student layer{j}->teacher layer{k}': attn[k]})

                std = torch.std(attn.detach())
                mu = torch.mean(attn.detach())
                cv = std/mu
                # print(f'cv block{i}:student layer{j}->teacher layer{k}',cv)

                loss = self.kd_loss(x_student,x_teacher)*cv
                loss_dict.update({f'loss_{i}': loss})

        return loss_dict

    # def forward(self, soft, pred, losses,gt_semantic_seg=None):
    #     for i in range(len(pred)):
    #         adaptor = self.adaptors[i]

    #         if self.use_attn:
    #             pred[i], soft[i], attn = adaptor(pred[i], soft[i])
    #             for j in range(attn.shape[0]):
    #                 losses.update({'attn' + str(i) + 'layer' + str(j): attn[j].clone()})
    #         else:
    #             pred[i],soft[i] = adaptor(pred[i],soft[i][0],gt_semantic_seg)

    #         if self.strategy == 'equal':
    #             loss = self.weights[i] * self.kd_loss(pred[i], soft[i])
    #             name = self.layers[i]
    #             losses.update({'loss_' + name: loss})
    #         elif self.strategy == 'self_adjust':
    #             loss = (1 / (self.weights[0] ** 2)) * \
    #                     self.kd_loss(pred[i], soft[i]) \
    #                     + torch.log(self.weights[0])
    #             name = self.layers[i]
    #             losses.update({'loss_' + name: loss})
    #             losses.update({'weight_' + name: self.weights[0]})

    #     if self.strategy == 'equal':
    #         pass
    #     elif self.strategy == 'self_adjust':
    #         losses['decode.loss_seg'] = (1 / (self.weights[1] ** 2)) * losses['decode.loss_seg'] + torch.log(
    #             self.weights[1])
    #         losses['aux.loss_seg'] = (1 / (self.weights[2] ** 2)) * losses['aux.loss_seg'] + torch.log(self.weights[2])

    #         losses.update({'weight_' + 'decode.loss_seg': self.weights[1]})
    #         losses.update({'weight_' + 'aux.loss_seg': self.weights[2]})
    #     return losses