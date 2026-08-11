import time, os, torch,copy
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from lib.timer import Timer, AverageMeter
from lib.utils import Logger,validate_gradient

from tqdm import tqdm
import torch.nn.functional as F
import gc


class Trainer(object):
    def __init__(self, args):
        self.config = args
        self.fre = args.fre
        # parameters
        self.start_epoch = 1
        self.max_epoch = args.max_epoch
        self.save_dir = args.save_dir
        self.device = args.device
        self.verbose = args.verbose
        self.max_points = args.max_points

        self.model = args.model.to(self.device)
        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        self.scheduler_freq = args.scheduler_freq
        self.snapshot_freq = args.snapshot_freq
        self.snapshot_dir = args.snapshot_dir 
        self.benchmark = args.benchmark
        self.iter_size = args.iter_size
        self.verbose_freq= args.verbose_freq

        self.w_circle_loss = args.w_circle_loss
        self.w_overlap_loss = args.w_overlap_loss
        self.w_saliency_loss = args.w_saliency_loss 
        self.desc_loss = args.desc_loss

        self.best_loss = 1e5
        self.best_recall = -1e5
        self.writer = SummaryWriter(log_dir=args.tboard_dir)
        self.logger = Logger(args.snapshot_dir)
        self.logger.write(f'#parameters {sum([x.nelement() for x in self.model.parameters()])/1000000.} M\n')
        

        if (args.pretrain !=''):
            self._load_pretrain(args.pretrain)
        
        self.loader =dict()
        self.loader['train']=args.train_loader
        self.loader['val']=args.val_loader
        self.loader['test'] = args.test_loader

        with open(f'{args.snapshot_dir}/model','w') as f:
            f.write(str(self.model))
        f.close()
 
    def _snapshot(self, epoch, name=None):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'best_recall': self.best_recall
        }
        if name is None:
            filename = os.path.join(self.save_dir, f'model_{epoch}.pth')
        else:
            filename = os.path.join(self.save_dir, f'model_{name}.pth')
        self.logger.write(f"Save model to {filename}\n")
        torch.save(state, filename)

    # load transformer except in_proj
    # load self-attention
    def load_transformer(self, model2_dict):
        state = torch.load('/home/tata/OverlapPredator_at_ex_1024/weights/geotransformer-3dmatch.pth.tar')
        model2_dict['gnn.rde_proj.weight'] = state['model']['transformer.rde_proj.weight']
        model2_dict['gnn.rde_proj.bias'] = state['model']['transformer.rde_proj.bias']
        model2_dict['gnn.rae_proj.weight'] = state['model']['transformer.rae_proj.weight']
        model2_dict['gnn.rae_proj.bias'] = state['model']['transformer.rae_proj.bias']
        model2_dict['gnn.embedding.div_term'] = state['model']['transformer.embedding.div_term']
        """ weight = {}
        for k,v in state['model'].items():
            if 'transformer' in k  and 'transformer.in_proj.weight' not in k and 'transformer.in_proj.bias' not in k and 'transformer.out_proj.weight' not in k and 'transformer.out_proj.bias' not in k:
                if '.1' in k or '.2' in k:
                    k = k[:31] + str(int(k[31])+1) + k[32:]
                    weight[k] = v
                elif '.3' in k or '.4' in k:
                    k = k[:31] + str(int(k[31])+2) + k[32:]
                    weight[k] = v
                elif '.5' in k:
                    k = k[:31] + str(int(k[31])+3) + k[32:]
                    weight[k] = v
                else:
                    weight[k] = v
        
        model2_dict.update(weight)
        weight = {}
        for k,v in model2_dict.items():
            if 'gnn.layers.0' in k:
                weight[k] = v
        
        for k,v in weight.items():
            model2_dict['transformer.transformer.layers.1.' + k[13:]] = v
            model2_dict['transformer.transformer.layers.4.' + k[13:]] = v
            model2_dict['transformer.transformer.layers.7.' + k[13:]] = v

        return weight """

    def _load_pretrain(self, resume):
        if os.path.isfile(resume):
            state = torch.load(resume)
            ############
            model2_dict = self.model.state_dict()
            state_dict = {k:v for k,v in state['state_dict'].items() if k in model2_dict.keys()}
            model2_dict.update(state_dict)
            ################# 第三章 ##############
            # self.load_transformer(model2_dict) # 单独训练GSE需要注释这里
            ################ 第三章 ################
            self.model.load_state_dict(model2_dict)
            """ edgescore_model2_dict = self.model.gnn.layers._modules['0'].edgescore.state_dict()
            edgescore_state = torch.load('/home/tata/OverlapPredator/snapshot/indoor/checkpoints/model_best_loss_edge_score.pth')
            edgescore_state_dict = {k:v for k,v in edgescore_state['state_dict'].items() if k in edgescore_model2_dict.keys()}
            edgescore_model2_dict.update(edgescore_state_dict)
            self.model.gnn.layers._modules['0'].edgescore.load_state_dict(edgescore_model2_dict)
            self.model.gnn.layers._modules['2'].edgescore.load_state_dict(edgescore_model2_dict)
            for para in self.model.gnn.layers._modules['0'].edgescore.parameters():
                para.requires_grad = False
            for para in self.model.gnn.layers._modules['2'].edgescore.parameters():
                para.requires_grad = False """
            ############
            self.start_epoch = state['epoch']
            self.scheduler.load_state_dict(state['scheduler'])
            ############
            model2_dict1 = self.optimizer.state_dict()
            state_dict1 = {k:v for k,v in state['optimizer'].items() if k in model2_dict1.keys()}
            model2_dict1.update(state_dict1)
            #self.optimizer.load_state_dict(model2_dict1)
            self.best_loss = state['best_loss']
            self.best_recall = state['best_recall']
            
            self.logger.write(f'Successfully load pretrained model from {resume}!\n')
            self.logger.write(f'Current best loss {self.best_loss}\n')
            self.logger.write(f'Current best recall {self.best_recall}\n')
        else:
            raise ValueError(f"=> no checkpoint found at '{resume}'")

    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']

    def stats_dict(self):
        stats=dict()
        stats['circle_loss']=0.
        stats['recall']=0.  # feature match recall, divided by number of ground truth pairs
        stats['saliency_loss'] = 0.
        stats['saliency_recall'] = 0.
        stats['saliency_precision'] = 0.
        stats['overlap_loss'] = 0.
        stats['overlap_recall']=0.
        stats['overlap_precision']=0.
        return stats

    def stats_meter(self):
        meters=dict()
        stats=self.stats_dict()
        for key,_ in stats.items():
            meters[key]=AverageMeter()
        return meters


    def inference_one_batch(self, inputs, phase):
        assert phase in ['train','val','test']
        ##################################
        # training
        if(phase == 'train'):
            self.model.train()
            ###############################################
            # forward pass
            feats, scores_overlap, scores_saliency, loss = self.model(inputs)  #[N1, C1], [N2, C2]
            pcd = inputs['points'][0]
            coords = inputs['points'][-1]
            len_src = inputs['stack_lengths'][0][0]
            c_rot, c_trans = inputs['rot'], inputs['trans']
            correspondence = inputs['correspondences']

            src_pcd, tgt_pcd = inputs['src_pcd_raw'], inputs['tgt_pcd_raw']
            src_feats, tgt_feats = feats[:len_src], feats[len_src:]

            ###################################################
            # get loss
            stats= self.desc_loss(src_pcd, tgt_pcd, src_feats, tgt_feats,correspondence, c_rot, c_trans, scores_overlap, scores_saliency, loss)

            c_loss = stats['circle_loss'] * self.w_circle_loss + stats['overlap_loss'] * self.w_overlap_loss + stats['saliency_loss'] * self.w_saliency_loss

            c_loss.backward()

        else:
            self.model.eval()
            with torch.no_grad():
                ###############################################
                # forward pass
                feats, scores_overlap, scores_saliency, loss = self.model(inputs)  #[N1, C1], [N2, C2]
                pcd =  inputs['points'][0]
                len_src = inputs['stack_lengths'][0][0]
                c_rot, c_trans = inputs['rot'], inputs['trans']
                correspondence = inputs['correspondences']

                src_pcd, tgt_pcd = inputs['src_pcd_raw'], inputs['tgt_pcd_raw']
                src_feats, tgt_feats = feats[:len_src], feats[len_src:]

                ###################################################
                # get loss
                stats= self.desc_loss(src_pcd, tgt_pcd, src_feats, tgt_feats,correspondence, c_rot, c_trans, scores_overlap, scores_saliency, loss)


        ##################################        
        # detach the gradients for loss terms
        stats['circle_loss'] = float(stats['circle_loss'].detach())
        stats['overlap_loss'] = float(stats['overlap_loss'].detach())
        stats['saliency_loss'] = float(stats['saliency_loss'].detach())
        
        return stats


    def inference_one_epoch(self,epoch, phase):
        gc.collect()
        assert phase in ['train','val','test']

        # init stats meter
        stats_meter = self.stats_meter()

        num_iter = int(len(self.loader[phase].dataset) // self.loader[phase].batch_size)
        c_loader_iter = self.loader[phase].__iter__()
        
        self.optimizer.zero_grad()
        for c_iter in tqdm(range(num_iter)): # loop through this epoch   
            ##################################
            # load inputs to device.
            inputs = c_loader_iter.next()
            for k, v in inputs.items():  
                if type(v) == list:
                    inputs[k] = [item.to(self.device) for item in v]
                elif type(v) == dict:
                    pass
                elif type(v) == str:
                    pass
                else:
                    inputs[k] = v.to(self.device)
            try:
                ##################################
                # forward pass
                # with torch.autograd.detect_anomaly():
                stats = self.inference_one_batch(inputs, phase)
                
                ###################################################
                # run optimisation
                if((c_iter+1) % self.iter_size == 0 and phase == 'train'):
                    gradient_valid = validate_gradient(self.model)
                    if(gradient_valid):
                        self.optimizer.step()
                    else:
                        self.logger.write('gradient not valid\n')
                    self.optimizer.zero_grad()
                
                ################################
                # update to stats_meter
                for key,value in stats.items():
                    stats_meter[key].update(value)
            except Exception as inst:
                print(inst)
            
            torch.cuda.empty_cache()
            
            if (c_iter + 1) % self.verbose_freq == 0 and self.verbose:
                curr_iter = num_iter * (epoch - 1) + c_iter
                for key, value in stats_meter.items():
                    self.writer.add_scalar(f'{phase}/{key}', value.avg, curr_iter)
                
                message = f'{phase} Epoch: {epoch} [{c_iter+1:4d}/{num_iter}]'
                for key,value in stats_meter.items():
                    message += f'{key}: {value.avg:.2f}\t'

                self.logger.write(message + '\n')

        message = f'{phase} Epoch: {epoch}'
        for key,value in stats_meter.items():
            message += f'{key}: {value.avg:.2f}\t'
        self.logger.write(message+'\n')

        return stats_meter


    def train(self):
        print('start training...')
        for epoch in range(self.start_epoch, self.max_epoch):
            self.inference_one_epoch(epoch,'train')
            self.scheduler.step()
            
            stats_meter = self.inference_one_epoch(epoch,'val')
            
            if stats_meter['circle_loss'].avg < self.best_loss:
                self.best_loss = stats_meter['circle_loss'].avg
                self._snapshot(epoch,'best_loss')
            if stats_meter['recall'].avg > self.best_recall:
                self.best_recall = stats_meter['recall'].avg
                self._snapshot(epoch,'best_recall')
            #### 修改参数保存名字 #####
            # self._snapshot(epoch,'temp_save_train_all_a' + str(self.fre) + str(epoch))
            self._snapshot(epoch,'tatanew_combine_base2' + str(self.fre) +'_'+ str(epoch))
            #self._snapshot(epoch,'temp_save')
            #### 修改参数保存名字 #####
            # we only add saliency loss when we get descent point-wise features
            if(stats_meter['recall'].avg>0.3):
                self.w_saliency_loss = 1.
            else:
                self.w_saliency_loss = 0.
                    
        # finish all epoch
        print("Training finish!")


    def eval(self):
        print('Start to evaluate on validation datasets...')
        stats_meter = self.inference_one_epoch(0,'val')
        
        for key, value in stats_meter.items():
            print(key, value.avg)
