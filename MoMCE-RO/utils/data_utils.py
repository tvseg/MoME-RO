# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import re

import numpy as np
import torch

from monai import data, transforms #data
import utils.dataset as custom_data
import copy
from glob import glob
import torch.nn.functional as F
import pandas as pd
import torch.nn as nn

class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def datafold_read_(args, report=None, dir=None, test_mode=None):

    data_dir_list = glob(dir + "**/data.nii.gz")
    data_dir_list.sort()
    
    # load data and report
    data = {}
    data_val_dir = []
    count = 0

    for f in data_dir_list:
        d = {}
        flag_test = False
        
        report_key = f.split('/')[-2]
        # print(report_key)
        # if args.data_dir.find('PCG') >=0:
        #     report_key = report_key.split('_')[-2].split('/')[-1]

        try:
            rep = report[report_key]
            if (test_mode <= 1):
                if '<test>' in rep:
                    if test_mode == 1:
                        flag_test = True 
                    else:
                        continue
                    rep = rep.replace('<test>', '')
                elif '<val>' in rep:
                    if test_mode == 0:
                        flag_test = True 
                    rep = rep.replace('<val>', '')
                elif '<train>' in rep:
                    rep = rep.replace('<train>', '')
                    count += 1
                else:
                    continue
                
            else:
                count += 1
                
            if rep.find('outlier') >= 0:
                print("outlier: ", f, report[report_key])
                continue
            
            d["report"] = rep
            d["side"] = d["report"].split(' side')[0].split(' ')[-1]

        except:
            continue

        # if (args.target >= 1) & (args.test_mode == 0):# & (args.stage == 1):
        #     if args.target == 4:
        #         d['image_add'] = f.replace('data.nii', 'data_add_reg.nii') #data_add_reg
        #     else:
        #         d['image_add'] = f.replace('data.nii', 'data_add.nii')
        #     try:
        #         if args.target != 4:
        #             glob(d['image_add'].replace('data_add', 'data_add_reg'))[0] #
        #         else:
        #             glob(d['image_add'])[0] #
        #     except:
        #         continue

        d['image'] = [f]
        d['label'] = f.replace('data.nii', 'label_gtv.nii')
        if args.target >= 2:
            eval_tag = dir.split('/')[-2]
            train_tag = '_trainset' if test_mode == 0 else ''
            d['gtv'] = ['%s/raw_%s%s/%s.nii'%(args.gtv_dir, eval_tag, train_tag, f.replace(dir, '').replace('/data.nii.gz', ''))] 
            
        d['id'] = report_key
        d['test_mode'] = max(test_mode-1, 0) 

        if d["side"] not in data.keys():
            data[d["side"]] = [] 

        if flag_test:
            data_val_dir.append(d)
        else:
            data[d["side"]].append(d)

    print(count)

    # split dataset
    tr = []
    val = []
    for side in data.keys():

        p_tr = 0.875
        p_val = 1
        if args.flag_pc:
            if args.stage < 3:
                tr += data[side][:int(p_tr*len(data[side])*args.p_data)] 
            if args.stage == 3:
                if test_mode == 1:
                    tr = data_val_dir[:5] #10 15
                    val = data_val_dir[15:20]
                elif test_mode == 0:
                    tr += data[side][:int(p_tr*len(data[side])*args.p_data)] 
                    val += data[side][int(p_tr*len(data[side])):int(p_val*len(data[side]))]
                elif test_mode > 1:
                    tr += data[side][:1] #2 3
                    val += data[side][3:4]
            else:
                if test_mode == 1:
                    val = data_val_dir[20:]
                elif test_mode == 0:
                    val += data[side][int(p_tr*len(data[side])):int(p_val*len(data[side]))]
                else:
                    val += data[side][4:]

        else:
            if args.test_mode >= 2: #extern
                val += data[side]
                tr += data[side]
            else: # args.test_mode =< 1:
                val = data_val_dir
                tr += data[side][:int(len(data[side])*args.p_data)] 

        print(side)
        print(">>>>>>> trainset: ", len(tr))
        print(">>>>>>> valset: ", len(val))

    if (args.flag_pc) & (args.test_mode == 3):
        data_dir = glob("/home/gpuadmin/yujin/ro-llama/prostate_target_volume/GN_added/**/data.nii.gz")
        for f in data_dir:
            d = {}
            report_key = f.split('/')[-2]

            rep = ''
            # rep = report[report_key]
            # if report[report_key].find('outlier') >= 0:
            #     print("outlier: ", f, report[report_key])
            #     continue

            d["report"] = rep
            if args.target >= 2:
                eval_tag = dir.split('/')[-2]
                train_tag = '_trainset' if test_mode == 0 else ''
                d['gtv'] = ['%s/raw_%s%s/%s.nii'%(args.gtv_dir, eval_tag, train_tag, f.replace(dir, '').replace('/data.nii.gz', ''))] 
                
            d['test_mode'] = max(test_mode-1, 0) 
            d['image'] = [f]
            d['label'] = f.replace('data.nii', 'label_gtv.nii')
            d['id'] = report_key
            val.append(d)

    # [print(v['id'], sep=', ') for v in val]

    return tr, val


def get_loader(args, retriever=None):
    report = {}
    datalist_ = []
    vallist_ = []

    # dataset
    if isinstance(args.report_dir, list):
        for test_mode, report_dir in enumerate(args.report_dir):

            if args.stage != 3:
                test_mode = args.test_mode
                data_dir = args.data_dir[0]
            else:
                data_dir = args.data_dir[test_mode]
                
            try:
                report_all = pd.read_excel(report_dir)
            except:
                report_all = pd.read_csv(report_dir)
            report.update(build_prompt(report_all, args))

            datalist, val_files = datafold_read_(args, report=report, dir=data_dir, test_mode=test_mode)
            if (args.stage == 3) & (test_mode >= 0): # 0:All 1:MoE 3:GN 
                datalist_ += datalist
                vallist_ += val_files

        print(">>>>>>> context: ", args.context)
        if args.stage == 3:
            datalist = datalist_
            val_files = vallist_ 
        
        if (len(datalist) == 0):
            datalist = val_files
    
    print(">>>>>>> trainset: ", len(datalist))
    print(">>>>>>> valset: ", len(val_files))

    # tokenize
    if args.context:
        if (args.compare_mode == 1) :
            retriever.text_encoder.cuda()

        if retriever.text_encoder.llm:
            for i, datalist_orig in enumerate([datalist, val_files]):
                for j, data_i in enumerate(datalist_orig): 

                    try:
                        
                        if (args.compare_mode == 1) :

                            inputs = retriever.tokenizer.encode_plus(
                                data_i['report'],
                                None,
                                add_special_tokens=True,
                                max_length=retriever.max_length,
                                padding= 'max_length',
                                truncation='longest_first',
                                return_token_type_ids=True
                            )
                            ids = inputs['input_ids']
                            mask = inputs['attention_mask']
                            ids = torch.tensor(ids).cuda().unsqueeze(0)
                            mask = torch.tensor(mask).cuda().unsqueeze(0)
                            
                            encoder_output = retriever.text_encoder.encoder(input_ids=ids, attention_mask=mask, return_dict=True)
                            pooled_sentence = encoder_output.last_hidden_state
                            tok_txt_ = pooled_sentence.detach().cpu().squeeze(0)

                        else:

                            # print(data_i['report'])
                            tok_txt_ = retriever.tokenizer.encode(data_i['report'])
                            m = nn.ConstantPad1d((0, max(0, retriever.max_length - retriever.context_length - len(tok_txt_))), 0)
                            tok_txt_ = m(torch.tensor(tok_txt_, dtype=torch.long))

                            if len(tok_txt_) > retriever.max_length - retriever.context_length:
                                # print('>>>>>>>>')
                                # print(data_i['report'])
                                tok_txt_ = torch.cat([tok_txt_[:retriever.max_length - retriever.context_length - 1], torch.tensor(retriever.tokenizer.vocab_size) if not args.flag_pc else torch.zeros(0)], dim=0)    
                                

                        data_i['raw_report'] = data_i['report']
                        data_i['report'] = tok_txt_

                    except:

                        # print('>>>>>>>>')
                        # print(data_i['report'])
                        if args.context_mode >= 2:
                            data_i['raw_report'] = data_i['report']
                        pass

    if args.target >= 3:
        key_together = ["image", "label", "gtv", "image_add"]
        key_together_woMR = ["image", "label", "gtv"] 
        if args.target == 4:
            key_together_woMR = key_together 
    elif args.target == 2:
        key_together = ["image", "label", "gtv"]
        key_together_woMR = ["image", "label", "gtv"]
    else:
        key_together = ["image", "label"]
        key_together_woMR = ["image", "label"]

    if (args.compare_mode == 1):# & (args.stage == 3):
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=key_together, image_only=False),
                transforms.EnsureChannelFirstd(keys=key_together, channel_dim='no_channel'),
                transforms.Orientationd(keys=key_together, axcodes="RAS"),
                # transforms.Spacingd(
                #     keys=key_together, pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
                # ),
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),
                transforms.ScaleIntensityRanged(
                    keys=["label"], a_min=0, a_max=args.c_max, b_min=0, b_max=args.c_max, clip=True, dtype=np.int8
                ),
                
                transforms.CropForegroundd(keys=key_together_woMR, source_key="image", allow_smaller=True),
                transforms.SpatialPadd(keys=key_together_woMR, spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
                # transforms.Resized(keys=key_together, spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
                transforms.Rotate90d(keys=key_together),
                transforms.RandCropByPosNegLabeld(
                    keys=key_together_woMR,
                    label_key="label",
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    pos=1,
                    neg=0.1,
                    num_samples=args.sw_batch_size, #1 #4
                    image_key="image",
                    image_threshold=0,
                ),
                # transforms.RandFlipd(keys=key_together, prob=args.RandFlipd_prob, spatial_axis=0),
                # transforms.RandFlipd(keys=key_together, prob=args.RandFlipd_prob, spatial_axis=1),
                # transforms.RandFlipd(keys=key_together, prob=args.RandFlipd_prob, spatial_axis=2),
                # transforms.RandRotate90d(keys=key_together, prob=args.RandRotate90d_prob, max_k=3),
                transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
                transforms.AsDiscreteD(keys=["label"], argmax=False, to_onehot=2),
                transforms.ToTensord(keys=key_together),
            ]
        )
        
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=key_together, image_only=False),
                transforms.EnsureChannelFirstd(keys=key_together, channel_dim='no_channel'),
                transforms.Orientationd(keys=key_together, axcodes="RAS"),
                # transforms.Spacingd(
                #     keys=key_together, pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
                # ),
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),
                transforms.ScaleIntensityRanged(
                    keys=["label"], a_min=0, a_max=args.c_max, b_min=0, b_max=args.c_max, clip=True, dtype=np.int8
                ),
                transforms.CropForegroundd(keys=key_together_woMR, source_key="image", allow_smaller=True),
                transforms.SpatialPadd(keys=key_together_woMR, spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
                # transforms.Resized(keys=key_together, spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
                transforms.Rotate90d(keys=key_together_woMR),
                # transforms.SpatialPadd(keys=key_together, spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
                transforms.AsDiscreteD(keys=["label"], argmax=False, to_onehot=2),
                transforms.ToTensord(keys=key_together),
            ]
        )
    
    else:
        
        train_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=key_together, image_only=False),
                transforms.EnsureChannelFirstd(keys=key_together, channel_dim='no_channel'),
                # transforms.EnsureChannelFirstd(keys=["image"], channel_dim='no_channel'),
                # transforms.EnsureChannelFirstd(keys=["label"], channel_dim=3),
                transforms.Orientationd(keys=key_together, axcodes="RAS"),
                # transforms.Spacingd(
                #     keys=key_together, pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
                # ),
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),
                transforms.ScaleIntensityRanged(
                    keys=["label"], a_min=args.c_min, a_max=args.c_max, b_min=0, b_max=1, clip=True, dtype=np.int8
                ),
                
                transforms.CropForegroundd(keys=key_together_woMR, source_key="image", allow_smaller=True),
                transforms.SpatialPadd(keys=key_together, spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
                # transforms.Resized(keys=key_together, spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
                transforms.Rotate90d(keys=key_together),
                transforms.RandCropByPosNegLabeld(
                    keys=key_together_woMR,
                    label_key="label",
                    spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    pos=1,
                    neg=0.1,
                    num_samples=args.sw_batch_size, #1 #4
                    image_key="image",
                    image_threshold=0,
                ),
                # transforms.RandFlipd(keys=key_together, prob=args.RandFlipd_prob, spatial_axis=0),
                # transforms.RandFlipd(keys=key_together, prob=args.RandFlipd_prob, spatial_axis=1),
                # transforms.RandFlipd(keys=key_together, prob=args.RandFlipd_prob, spatial_axis=2),
                # transforms.RandRotate90d(keys=key_together, prob=args.RandRotate90d_prob, max_k=3),
                transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
                # transforms.AsDiscreteD(keys=["label"], threshold=1.5, to_onehot=args.c_max),
                transforms.ToTensord(keys=key_together),
            ]
        )
        
        val_transform = transforms.Compose(
            [
                transforms.LoadImaged(keys=key_together, image_only=False),
                transforms.EnsureChannelFirstd(keys=key_together, channel_dim='no_channel'),
                # transforms.EnsureChannelFirstd(keys=["image"], channel_dim='no_channel'),
                # transforms.EnsureChannelFirstd(keys=["label"], channel_dim=3),
                transforms.Orientationd(keys=key_together, axcodes="RAS"),
                # transforms.Spacingd(
                #     keys=key_together, pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
                # ),
                transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),
                transforms.ScaleIntensityRanged(
                    keys=["label"], a_min=args.c_min, a_max=args.c_max, b_min=0, b_max=1, clip=True, dtype=np.int8
                ),
                transforms.CropForegroundd(keys=key_together_woMR, source_key="image", allow_smaller=True),
                transforms.SpatialPadd(keys=key_together, spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
                # transforms.Resized(keys=key_together, spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
                transforms.Rotate90d(keys=key_together),
                # transforms.SpatialPadd(keys=key_together, spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
                # transforms.AsDiscreteD(keys=["label"], argmax=True, to_onehot=2),
                # transforms.AsDiscreteD(keys=["label"], threshold=1.5, to_onehot=args.c_max),
                transforms.ToTensord(keys=key_together),
            ]
        )
    
    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=key_together, image_only=False),
            transforms.EnsureChannelFirstd(keys=key_together, channel_dim='no_channel'),
            transforms.Orientationd(keys=key_together, axcodes="RAS"),
            # transforms.Spacingd(keys="image", pixdim=(args.space_x, args.space_y, args.space_z), mode="bilinear"),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.ScaleIntensityRanged(
                keys=["label"], a_min=args.c_min, a_max=args.c_max, b_min=0, b_max=1, clip=True, dtype=np.int8
            ),
            transforms.CropForegroundd(keys=key_together_woMR, source_key="image", allow_smaller=True),
            transforms.SpatialPadd(keys=key_together, spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
            # transforms.Resized(keys=key_together, spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
            transforms.Rotate90d(keys=key_together),
            # transforms.SpatialPadd(keys=key_together, spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
            # transforms.AsDiscreteD(keys=["label"], threshold=1.5, to_onehot=args.c_max),
            transforms.ToTensord(keys=key_together), 
        ]
    )

    # datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
    if args.use_normal_dataset:
        train_ds = custom_data.Dataset(data=datalist, transform=train_transform, report=report) #report
    else:
        train_ds = data.CacheDataset(
            data=datalist, transform=train_transform, cache_num=24, cache_rate=1.0, num_workers=args.workers
        )
    train_sampler = Sampler(train_ds) if args.distributed else None
    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        sampler=train_sampler,
        pin_memory=False,
    )
    
    if args.test_mode & (args.stage != 3):
        # test_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        test_ds = custom_data.Dataset(data=val_files, transform=test_transform, report=report) 
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            sampler=test_sampler,
            pin_memory=False,
            persistent_workers=True,
        )
        loader = [train_loader, test_loader]
    
    else:
        # val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        val_ds = custom_data.Dataset(data=val_files, transform=val_transform, report=report) 
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=False
        )
        loader = [train_loader, val_loader]

    return loader


def build_prompt(df, args):

    unit_no = list(df['Unit No'])
    unit_no_unique = np.unique(unit_no)
    report = {}

    for no in unit_no_unique:

        row = df.loc[df['Unit No'] == no]
        if unit_no.count(no) == 1:
            text_prompt = prepare_report(args, row, 0)
        else:
            if args.test_mode == 2:
                text_prompt = None
            else:
                text_prompt_list = []
                for i in range(row.ndim):
                    text_prompt_list.append(prepare_report(args, row, i))
                if text_prompt_list[0] is not None:
                    # print(no, text_prompt_list)
                    text_prompt_list = np.unique(text_prompt_list) 
                    text_prompt = '; '.join(text_prompt_list)
                else:
                    text_prompt = None
            # text_prompt = text_prompt.replace('left','both').replace('rigt','both')

        if text_prompt is not None:
            report[str(no)] = text_prompt + (" <SEG>" if (args.compare_mode == 0) & (args.flag_pc != 1) else "") #(args.stage != 1)
        # else:
        #     print(no, text_prompt)
            
    return report


def prepare_report(args, row, i=0):

    if args.flag_pc:

        prompt = row['summary_prompt'].values[i]
        prompt_list = prompt.split('\n')
        if len(prompt_list) < 12:
            return None
        

        gs = prompt_list[2].lower()
        try: 
            if (gs.find('grade')<0):
                gs = prompt_list[3].lower().split(':')[-1]
                prompt_list[3] = prompt_list[2]
        except:
            gs = '?'
        try:
            gs = gs.split('->')[0]
        except:
            pass


        stage = prompt_list[3].split(':')[1]
        # print(stage)
        try:
            total_stage = stage.split('Stage ')[1]
            total_stage = total_stage.split(' ')[0]
        except:
            total_stage = '?'
        stage = stage.split(',')[0].replace(' ','')
        
        try:
            stage = stage.split('->')[0]
        except:
            pass
        try:
            t_stage, nm_stage = stage.split('N')
            n_stage, m_stage = (nm_stage).split('M')
            n_stage = ('N' + re.sub(r'[^0-9]', '', n_stage))
            m_stage = ('M' + re.sub(r'[^0-9]', '', m_stage))
        except:
            t_stage, n_stage, m_stage = stage[:-4], stage[-4:-2].replace('M','N'), stage[-2:].replace('N','M')
        if t_stage == '':
            t_stage, n_stage, m_stage = '?', '?', '?'
        if len(t_stage) > 6:
            t_stage = t_stage.split('N')[0]
        if len(n_stage) != 2:
            n_stage = 'N?'
        t_stage = t_stage.replace('Initial','')
        
        surgery = prompt_list[4].split(': ')[-1].lower()

        # meta = prompt_list[11].split(': ')[-1].lower()
        # meta_list = meta.split(' ')
        # meta_list_ = []
        # for sur in meta_list:
        #     if any([sur.find(cand)>=0 for cand in ['no', 'biochemical', 'local', 'regional', 'distant' , 'other', 'n/a']]):
        #         meta_list_.append(sur)
        # meta = ' and '.join(meta_list_)
        # meta = re.sub(r'[,.]', '', meta)
        # if meta == '':
        #     meta = 'yes'

        lne = prompt_list[10].lower().split(':')[-1]
        if lne[0] == ' ':
            lne = lne[1:]
        if lne.find('n/a')>=0:
            lne = 'none'

        psa = prompt_list[5].lower()

        age = row['Age'].values[i]


        # # trial1
        # text_prompt = ', '.join([t_stage, n_stage, gs, meta, orientation + ' side'])
        # text_prompt = text_prompt.replace('N/A', '?')  
         
        # # trial2
        # text_prompt = ', '.join([n_stage, gs, meta, orientation + ' side'])
        # text_prompt = text_prompt.replace('N/A', '?')  

        # # trial3
        # text_prompt = ', '.join([n_stage, gs, meta + ' metastasis', svi, lne, orientation + ' side'])

        # # trial4
        # text_prompt = ', '.join([gs, t_stage, n_stage, 'lymph node metastasis: ' + lne, 'age: %d'%age, psa])

        # # trial5
        # text_prompt = ', '.join([gs, t_stage, n_stage, 'lymph node metastasis: ' + lne, 'age: %d'%age])

        # # trial6
        # # print(psa)
        try:
            psai = psa.split('initial ')[1].split(' ')[0]
            if psai == '':
                psai = '?'
        except:
            psai = '?'
        # text_prompt = ', '.join([gs, t_stage, n_stage, 'lymph node metastasis: ' + lne, 'age: %d'%age, 'specific antigen value: ' + psai])

        # print(psai)
        psai = re.sub(r'[^0-9.]', '', psai)
        # print(psai)
        
        try:
            # if args.meta:
            #     psai = float(psai)
            #     psai = '%3.1f'%(psai)
            # else:
            psai = int(float(psai))
            psai = '%d'%(4 if psai > 30 else 3 if psai > 20 else 2 if psai > 10 else 1 if psai > 5 else 0)
        except:
            psai = '?'

        gs = gs.split(')')[0] + ')'
        gs = gs.replace('gelason score ', '')

        text_prompt = ' '.join([gs.replace('grade:', '<grade>'), '<stage> %s, %s'%(t_stage, n_stage), '<metastasis> %s'%('+' if lne == 'positive' else '-'), '<age> %d'%age, '<psa> ' + psai])
        if text_prompt.find('<grade>') < 0:
            text_prompt = '<grade>' + text_prompt

        # if args.meta:
        #     text_prompt = text_prompt + ', postactomy : %s'%(surgery)
        
        # trial7
        # text_prompt = ', '.join([gs, 'Stage: '+total_stage, t_stage, n_stage, 'lymph node metastasis: ' + lne, 'age: %d'%age, 'specific antigen value: ' + psai])

        text_prompt = text_prompt.replace('N/A', '?').replace(' gleason score', '')
        text_prompt = text_prompt.replace('n/a', '?') 

        # print(text_prompt)
        return text_prompt
    
    else:
    
        try:
            
            try:
                age = int(row['Age'].values[i])
            except:
                age = 'unknown'
            sex = row['Sex'].values[i]
            if sex == 'F':
                sex = 'female'
            elif sex == 'M':
                sex = 'male'
            
            pathology = row['Pathology'].values[i]
            if pathology == 'IDC':
                pathology = 'Invasive ductal ca'


            if args.test_mode >= 2:
                
                t_stage = 'c' + str(row['icT'].values[i]) 
                n_stage = '' + str(row['icN'].values[i]) 
                m_stage = '' + str(row['icM'].values[i])
                if n_stage == 'nan':
                    print(row['icN'].values[i])
                    n_stage == 'unknown'
                if t_stage == 'cnan':
                    t_stage == 'unknown'
                t_stage = t_stage.replace(' or ', '/').replace(' ', '')
                
                if n_stage.find('2023') >= 0:
                    time = n_stage.split('-')
                    n_stage = 'N' + str(int(time[1])) + '/' + str(int(time[2].split(' ')[0])) 

                subsite = row['Subsite 1'].values[i]
                if 'Right' in subsite or 'right' in subsite or 'Rt' in subsite:
                    orientation = 'right'
                elif 'Left' in subsite or 'left' in subsite or 'Lt' in subsite:
                    orientation = 'left'
                elif 'Both' in subsite or 'both' in subsite:
                    orientation = 'both'
                    return None
                else:
                    orientation = 'unknown' #raise NotImplementedError()

                remark = row['Remark'].values[i]
                if 'BCS' in remark:
                    surgery = 'breast conserving surgery'
                elif 'Postop' in remark:
                    surgery = 'total mastectomy surgery'
                else:
                    surgery = 'unknown type surgery'#raise NotImplementedError()
            
            else: 
            
                t_stage = 'cT' + str(row['icT'].values[i]) 
                n_stage = 'N' + str(row['icN'].values[i]) 
                m_stage = 'M' + str(row['icM'].values[i])
                if n_stage == 'Nnan':
                    # print(row['icN'].values[i])
                    n_stage == 'unknown'
                if t_stage == 'cTnan':
                    t_stage == 'unknown'
                t_stage = t_stage.replace(' or ', '/').replace(' ', '')
                # c_stage = t_stage + n_stage + m_stage

                if n_stage.find('2023') >= 0:
                    time = n_stage.split('-')
                    n_stage = 'N' + str(int(time[1])) + '/' + str(int(time[2].split(' ')[0])) 

                surgery_types = row['Types'].values[i]
                if surgery_types == 'BCS':
                    surgery = 'breast conserving surgery'
                elif surgery_types == 'mastectomy':
                    surgery = 'total mastectomy surgery'
                else:
                    surgery = 'unknown type surgery'#raise NotImplementedError()

                subsite = row['Subsite 1'].values[i]
                if 'Right' in subsite or 'right' in subsite or 'Rt' in subsite:
                    orientation = 'right'
                elif 'Left' in subsite or 'left' in subsite or 'Lt' in subsite:
                    orientation = 'left'
                else:
                    return None
                    orientation = 'unknown' #raise NotImplementedError()
            
            if args.ablation == "_n3":
                n_stage = 'N3' 
            elif args.ablation == "_n0":
                n_stage = "N0"

            if args.ablation == "_t4":
                t_stage = 'cT3' 
            elif args.ablation == "_t1":
                t_stage = "cT1"

            if args.ablation == "_surgery":
                if surgery == 'breast conserving surgery':
                    surgery = 'total mastectomy surgery'
                elif surgery == 'total mastectomy surgery':
                    surgery = 'breast conserving surgery'

            if args.context_mode == 0:
                # text_prompt = ', '.join([n_stage, surgery, orientation + ' side']) 
                text_prompt = ', '.join([n_stage, surgery, orientation + ' side'])
            elif args.context_mode == 1:
                if args.ablation == '_swap':
                    text_prompt = ', '.join([surgery, n_stage, t_stage, orientation + ' side'])
                else:
                    if args.ablation == '_omitN':
                        text_prompt = ', '.join([t_stage, surgery, orientation + ' side'])
                    elif args.ablation == '_omitT':
                        text_prompt = ', '.join([n_stage, surgery, orientation + ' side'])
                    elif args.ablation == '_omitSurgery':
                        text_prompt = ', '.join([t_stage, n_stage, orientation + ' side'])
                    elif args.ablation == '_omitLaterality':
                        text_prompt = ', '.join([t_stage, n_stage, surgery])
                    else:
                        text_prompt = ', '.join([n_stage, t_stage, surgery, orientation + ' side'])
                # text_prompt += str(t_stage)
            elif args.context_mode == 2:
                try:
                    n_stage = int(n_stage[1])
                except:
                    n_stage = '?'
                # if t_stage[2] == 'i':
                #     t_stage = 'i'
                # else:
                try:
                    t_stage = int(t_stage[2])
                except:
                    t_stage = '?'
                try:
                    surgery = int(0 if surgery == 'breast conserving surgery' else (1 if surgery == 'mastectomy' else '?'))
                except:
                    surgery = '?'
                try:
                    orientation = int(0 if orientation == 'left' else (1 if orientation == 'right' else '?'))
                except:
                    orientation = '?'

                if args.ablation == '_omitN':
                    n_stage = '?'
                elif args.ablation == '_omitT':
                    t_stage = '0'
                elif args.ablation == '_omitSurgery':
                    surgery = '?'
                elif args.ablation == '_omitLaterality':
                    orientation = '?'
                
                text_prompt = '%s%s%s%s'%(str(n_stage), str(t_stage), str(surgery), str(orientation))
                # print(text_prompt)
                
            # text_prompt = 'Target volume for {} years old {} patient who underwent {} and pathologically confirmed as {}. The tumor was located in {} side. {} {}'.format(age, sex, surgery, pathology, orientation, chemotherapy, stage)

            return text_prompt

        except:
            
            print(str(row['icT'].values[i]))
            return None
    

def prepare_report_rogpt(context):
    
    pre_context = context
    check_kr = re.compile('[|가-힣]+').findall(pre_context)
    if (len(check_kr)>0):
        return None
    pre_context = pre_context.lower()
    
    try:
        # orientation
        if any([key in pre_context for key in ['right', 'rt.', 'r.']]):
            ori = 'right'
        elif any([key in pre_context for key in ['left', 'lt.', 'l.']]):
            ori = 'left'
        elif any([key in pre_context for key in ['both', 'bt.']]):
            ori = 'both'
        else:
            ori = 'unknown'
            
        # margin
        wb, rn = False, False
        if any([key in pre_context for key in ['whole breast']]):
            wb = True
        if any([key in pre_context for key in ['regional',  'l/n', 'rni']]):
            rn = True
        mar = ''
        mar += 'whole' if wb else 'none'
        mar += ' + lymph' if rn else ' + none'
        
        # surgery 
        if any([key in pre_context for key in ['breast conserving surgery', 'breast conservation surgery', 'conserv', 'bcs']]):
            sur = 'breast conserving surgery'
        elif any([key in pre_context for key in ['chest wall', 'CW', 'total mastectomy', 'mrm', 'modified radical mastectomy']]):
            sur = 'total mastectomy surgery' 
        elif any([key in pre_context for key in ['pm', 'chest wall', 'CW']]):
            sur = 'partial mastectomy surgery' 
        else:
            sur = 'unknown type surgery'
        
        # therapy
        therapy = ''
        if any([key in pre_context for key in ['fast', 'forward']]):
            therapy += 'fast'
        else:
            therapy += 'none'
        if any([key in pre_context for key in ['sib', 'simultaneous', 'integrated']]):
            therapy += ' + boost'
        else:
            therapy += ' + none'
        ## gray
        dose = ''.join(pre_context[:pre_context.find('gy')].split(' ')[-3:])
        dose = ''.join(re.compile('[0-9.]').findall(dose))
        try: 
            dose = '%2.2f gy'%float(dose)
        except:
            dose = 'unknown gy'
        ## fraction
        fr = ''.join(pre_context[:pre_context.find('fx')].split(' ')[-2:])
        fr = ''.join(re.compile('[0-9.]').findall(fr))
        try: 
            fr = '%2.2f fx'%float(fr)
        except:
            fr = 'unknown fx'
        dose += ' / ' + fr
        therapy += ' ' + dose

        # anatomy
        ana = ''
        if any([key in pre_context for key in ['estro']]):
            ana += 'ESTRO'
        else:
            ana += 'none'
        if any([key in pre_context for key in ['rtog']]):
            ana += ' + RTOG'
        else:
            ana += ' + none'
        if any([key in pre_context for key in ['axilla', 'axillary', 'axl', 'ax']]):
            ana += ' + AXL'
        else:
            ana += ' + none'
        if any([key in pre_context for key in ['mammary', 'imn']]):
            ana += ' + IMN'
        else:
            ana += ' + none'
        if any([key in pre_context for key in ['supraclavicular', 'scl', 'scf']]):
            ana += ' + SCL'
        else:
            ana += ' + none'
        

        # context = 'Orientation: {} side; Margin: {}; Surgery: {}; Therapy: {}; Anatomy: {}'.format(ori, mar, sur, therapy, ana)

        # Surgery >> Negative
            
        # # OMA
        # context = 'Orientation: {} side; Margin: {}; Anatomy: {}'.format(ori, mar, ana)

        # OMAT
        context = 'Orientation: {} side; Margin: {}; Anatomy: {}; Therapy: {}'.format(ori, mar, ana, therapy)


        return context

    except:
        
        return None


def prepare_report_rogpt_v2(context):
    
    pre_context = context
    check_kr = re.compile('[|가-힣]+').findall(pre_context)
    if (len(check_kr)>0):
        return None
    
    pre_context = pre_context.lower()

    if pre_context == 'nan':
        return "outlier" + pre_context
        
    pre_context = pre_context.split('<dose>')[0]
    pre_context = pre_context.replace('\n',' ')

    try:
        # orientation
        if any([key in pre_context for key in ['right', 'rt.']]):
            ori = 'right'
        elif any([key in pre_context for key in ['left', 'lt.']]):
            ori = 'left'
        elif any([key in pre_context for key in ['both', 'bt.']]):
            ori = 'both'
        else:
            ori = 'unknown'
        context = pre_context.replace('right', '').replace('rt.', '').replace('left', '').replace('lt.', '').replace('both', '').replace('bt.', '')
        # context = 'Orientation: {} side; {} <SEG>'.format(ori, pre_context) 
        # print(context)
            
        # margin
        wb, rn = False, False
        if any([key in context for key in ['whole']]):
            wb = True
        if any([key in context for key in ['regional',  'l/n', 'rni']]):
            rn = True
        mar = ''
        mar += 'whole' if wb else 'none'
        mar += ' + lymph' if rn else ' + none'
        
    #     # surgery 
    #     if any([key in pre_context for key in ['breast conserving surgery', 'breast conservation surgery', 'conserv', 'bcs']]):
    #         sur = 'breast conserving surgery'
    #     elif any([key in pre_context for key in ['chest wall', 'CW', 'total mastectomy', 'mrm', 'modified radical mastectomy']]):
    #         sur = 'total mastectomy surgery' 
    #     elif any([key in pre_context for key in ['pm', 'chest wall', 'CW']]):
    #         sur = 'partial mastectomy surgery' 
    #     else:
    #         sur = 'unknown type surgery'
        
        # therapy
        therapy = ''
        if any([key in context for key in ['pbi']]):
            therapy += 'pbi'
        else:
            therapy += 'none'
        # if any([key in context for key in ['fast', 'forward']]):
        #     therapy += ' + fast'
        # else:
        #     therapy += ' + none'
        # if any([key in context for key in ['sib', 'simultaneous', 'integrated']]):
        #     therapy += ' + boost'
        # else:
        #     therapy += ' + none'
        # if any([key in context for key in ['imrt']]):
        #     therapy += ' + imrt'
        # else:
        #     therapy += ' + none'
        # if any([key in context for key in ['vmat']]):
        #     therapy += ' + vmat'
        # else:
        #     therapy += ' + none'
        # if any([key in context for key in ['pmrt', 'post']]): 
        #     therapy += ' + pmrt'
        # else:
        #     therapy += ' + none'
        # if any([key in context for key in ['fraction', 'hypo']]):
        #     therapy += ' + hypofraction'
        # else:
        #     therapy += ' + none'

        
    #     ## gray
    #     dose = ''.join(pre_context[:pre_context.find('gy')].split(' ')[-3:])
    #     dose = ''.join(re.compile('[0-9.]').findall(dose))
    #     try: 
    #         dose = '%2.2f gy'%float(dose)
    #     except:
    #         dose = 'unknown gy'
    #     ## fraction
    #     fr = ''.join(pre_context[:pre_context.find('fx')].split(' ')[-2:])
    #     fr = ''.join(re.compile('[0-9.]').findall(fr))
    #     try: 
    #         fr = '%2.2f fx'%float(fr)
    #     except:
    #         fr = 'unknown fx'
    #     dose += ' / ' + fr
    #     therapy += ' ' + dose

        # anatomy
        ana = ''
        if any([key in context for key in ['estro']]):
            ana += 'ESTRO'
        else:
            ana += 'none'
        if any([key in context for key in ['rtog']]):
            ana += ' + RTOG'
        else:
            ana += ' + none'
        if any([key in context for key in ['axilla', 'axillary', 'axl', 'ax']]):
            ana += ' + AXL'
        else:
            ana += ' + none'
        if any([key in context for key in ['mammary', 'imn']]):
            ana += ' + IMN'
        else:
            ana += ' + none'
        if any([key in context for key in ['supraclavicular', 'scl', 'scf']]):
            ana += ' + SCL'
        else:
            ana += ' + none'
        

    #     # context = 'Orientation: {} side; Margin: {}; Surgery: {}; Therapy: {}; Anatomy: {}'.format(ori, mar, sur, therapy, ana)

    #     # Surgery >> Negative
            
        post_context = 'Orientation: {} side; Margin: {}; Therapy: {}'.format(ori, mar, therapy)
            
        # OMA (RL Flipping: Pass)
        # context = 'Orientation: {} side; Margin: {}; Anatomy: {}'.format(ori, mar, ana)
        # post_context = 'Orientation: {} side; Margin: {}; Anatomy: {}; Therapy: '.format(ori, mar, ana, therapy)

        # OMAT
        # post_context = 'Orientation: {} side; Margin: {}; Anatomy: {}; Therapy: {}'.format(ori, mar, ana, therapy)

        # print(pre_context)
        # print('>> ' + post_context)
        # print('\n')

        return post_context

    except:
        
        return None
