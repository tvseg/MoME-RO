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

from monai import data, transforms 
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

        try:
            rep = report[report_key]
            count += 1
                
            if rep.find('outlier') >= 0:
                # print("outlier: ", f, report[report_key])
                continue
            
            d["report"] = rep
            # print(rep)
            d["side"] = d["report"].split('<psa> ')[1].split(' ')[0]

        except:
            continue

        d['image'] = [f]
        d['label'] = f.replace('data.nii', 'label_gtv.nii')
        if args.target >= 2:
            eval_tag = dir.split('/')[-2]
            train_tag = '_trainset' if test_mode == 0 else ''
            d['gtv'] = ['%s/raw_%s%s/%s.nii'%(args.gtv_dir, eval_tag, train_tag, f.replace(dir, '').replace('/data.nii.gz', ''))] 
            if len(glob(d['gtv'][0])) == 0:
                continue
            
        d['id'] = report_key
        d['test_mode'] = max(test_mode-1, 0) 
    
        if ((args.logdir.find('Finetune') >= 0) | (args.pretrained_dir.find('Finetune') >= 0)) & (test_mode == 4):
            d['test_mode'] = 1

        if d["side"] not in data.keys():
            data[d["side"]] = [] 

        if flag_test:
            data_val_dir.append(d)
        else:
            data[d["side"]].append(d)

    print(count)
    # if (dir.find('MGH') >= 0):
    #     [data["3"].append(i) for i in data.pop("4")]
    # print(data)

    # split dataset
    tr = []
    val = []
    for side in data.keys():

        if args.stage == 3:
            tr += data[side][:args.shot] 
            val += data[side][3:4]
        else:
            val += data[side][4:]

        print(side)
        print(">>>>>>> trainset: ", len(tr))
        print(">>>>>>> valset: ", len(val))

    [print(v['id'], sep=', ') for v in val]

    return tr, val


def get_loader(args, retriever=None):

    report = {}
    datalist_ = []
    vallist_ = []

    # dataset
    if isinstance(args.report_dir, list):
        for data_dir, report_dir in zip(args.data_dir, args.report_dir):

            test_mode = args.test_mode
            try:
                report_all = pd.read_excel(report_dir)
            except:
                report_all = pd.read_csv(report_dir)
            report.update(build_prompt(report_all, args))

            datalist, val_files = datafold_read_(args, report=report, dir=data_dir, test_mode=test_mode)
            datalist_ += datalist
            vallist_ += val_files

        print(">>>>>>> context: ", args.context)
        if (args.logdir.find('_save') >= 0):
            args.max_epochs = 2

        if args.stage == 3:
            datalist = datalist_
            val_files = vallist_ 
        
        if (len(datalist) == 0):
            datalist = val_files

    print(">>>>>>> trainset: ", len(datalist))
    print(">>>>>>> valset: ", len(val_files))

    # tokenize
    if args.context:

        if retriever.text_encoder.llm:
            for i, datalist_orig in enumerate([datalist, val_files]):
                for j, data_i in enumerate(datalist_orig): 

                    if 'raw_report' not in data_i.keys():
                        data_i['raw_report'] = data_i['report']

                    if args.meta:
                        try:
                            data_i['report'], _ = data_i['report'].split('<real>')
                        except:
                            pass

                    try:
                        tok_txt_ = retriever.tokenizer.encode(data_i['report'])
                        m = nn.ConstantPad1d((0, max(0, retriever.max_length - retriever.context_length - len(tok_txt_))), 0)
                        tok_txt_ = m(torch.tensor(tok_txt_, dtype=torch.long))

                        if len(tok_txt_) > retriever.max_length - retriever.context_length:
                            tok_txt_ = torch.cat([tok_txt_[:retriever.max_length - retriever.context_length - 1], torch.tensor(retriever.tokenizer.vocab_size) if not args.flag_pc else torch.zeros(0)], dim=0)    
                
                        data_i['report'] = tok_txt_

                    except:
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

    if (args.compare_mode == 1): 
        
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
                transforms.AsDiscreteD(keys=["label"], to_onehot=2),
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
                transforms.AsDiscreteD(keys=["label"], to_onehot=2),
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
    
    if (args.test_mode > 0) & (args.stage != 3):
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
        try:
            text_prompt = row['mgh_raw'].values[0]
            # print(text_prompt)
            if (text_prompt != text_prompt):
                continue
            if (row['Train'].values[0] == 0) | (text_prompt.find("?") >= 0):
                print('outlier')
                continue
            text_prompt, real = text_prompt.split(" <psa> ")
            if real[-1] == '.':
                real = real[:-1]
            psai = float(real)
            psai = '%d'%(4 if psai > 30 else 3 if psai > 20 else 2 if psai > 10 else 1 if psai > 5 else 0)
            text_prompt += (' <psa> ' + psai)
            if args.meta:
                text_prompt += (' <real>' + real)
            # print(text_prompt)
        except:
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

        if text_prompt is not None:
            report[str(no)] = text_prompt + (" <SEG>" if (args.compare_mode == 0) & (args.flag_pc != 1) else "") 
            
    return report


def prepare_report(args, row, i=0):

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

    lne = prompt_list[10].lower().split(':')[-1]
    if lne[0] == ' ':
        lne = lne[1:]
    if lne.find('n/a')>=0:
        lne = 'none'

    psa = prompt_list[5].lower()

    age = row['Age'].values[i]

    try:
        psai = psa.split('initial ')[1].split(' ')[0]
        if psai == '':
            psai = '?'
    except:
        psai = '?'

    psai = re.sub(r'[^0-9.]', '', psai)
    
    try:
        if args.meta:
            psai_real = float(psai)
            psai_real = '%3.1f'%(psai_real)
        psai = int(float(psai))
        psai = '%d'%(4 if psai > 30 else 3 if psai > 20 else 2 if psai > 10 else 1 if psai > 5 else 0)
        if args.meta:
            psai += ' <real>' + psai_real
    except:
        psai = '?'

    gs = gs.split(')')[0] + ')'
    gs = gs.replace('gelason score ', '')

    text_prompt = ' '.join([gs.replace('grade:', '<grade>'), '<stage> %s, %s'%(t_stage, n_stage), '<metastasis> %s'%('+' if lne == 'positive' else '-'), '<age> %d'%age, '<psa> ' + psai])
    if text_prompt.find('<grade>') < 0:
        text_prompt = '<grade>' + text_prompt
    
    if args.context_mode == 0:
        text_prompt += ' <prostectomy> %s'%prompt_list[4].lower().split(':')[-1].replace('.','')
        # print(text_prompt)

    text_prompt = text_prompt.replace('N/A', '?').replace(' gleason score', '')
    text_prompt = text_prompt.replace('n/a', '?') 

    return text_prompt

    