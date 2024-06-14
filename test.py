# coding=utf-8
# ----------------------- load package ----------------------- #
import os
import cv2
import sys
import datetime
import random 
import yaml
from yaml.loader import SafeLoader
from apex import amp
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

from utils import dataset
from net import pvt_v2_b2 as mynet
# ----------------------- load package ----------------------- #



# 原始的损失函数：
def total_loss(pred, mask):
    """
    wBCE损失和wIoU损失
    """
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)

    return (wbce + wiou).mean()

def bce(pred, mask):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    return bce.mean()

def bce_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    return wbce.mean()


def validate(model, val_loader):
    model.train(False)
    avg_mae = 0.0
    nums = 0
    with torch.no_grad():
        for image, mask, shape, name in val_loader:
            image, mask = image.cuda().float(), mask.cuda().float()
            out = model(image)
            pred = torch.sigmoid(out)
            avg_mae += torch.abs(pred - mask).mean()
            nums += 1

    model.train(True)
    return (avg_mae / nums).item()


def train(Dataset, Network, config_dict):
    ## Set random seeds
    seed = config_dict["Random_seeds"]
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    ## dataset
    cfg = Dataset.Config(datapath=config_dict["train_datapath"], savepath=config_dict["out_path"], 
                            mode='train', batch=config_dict["batch"], lr=config_dict["lr"], 
                            resize=config_dict["resize"], trainsize=config_dict["trainsize"],
                            momen=config_dict["momen"], decay=config_dict["decay"], epoch=config_dict["epoch"])
    data = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, num_workers=config_dict["train_num_workers"])
    ## val dataloader
    val_cfg = Dataset.Config(datapath=config_dict["val_datapath"], mode='test', resize=config_dict["resize"],)
    val_data = Dataset.Data(val_cfg)
    val_loader = DataLoader(val_data, batch_size=cfg.batch, shuffle=False, num_workers=config_dict["val_num_workers"])
    min_mae = 1.0
    best_epoch = 0
    ## network
    net = Network()
    net.load_state_dict(torch.load("res/pvt_v2_b2.pth"), strict=False)
    # net.load_from()
    
    start_epoch = 0 # 从第0个epoch开始
    if config_dict["continue_train"]:
        start_epoch = config_dict["continue_train_start_epoch"]
        net.load_state_dict(torch.load(config_dict["continue_train_pth"]))
    net.cuda()
    net.train(True)
    ## parameter
    enc_params, dec_params = [], []
    for name, param in net.named_parameters():
        if 'backbone' in name:
            enc_params.append(param)
        else:
            dec_params.append(param)

    optimizer = torch.optim.SGD([{'params': enc_params}, {'params': dec_params}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
    sw = SummaryWriter(cfg.savepath)
    global_step = 0

    for epoch in range(start_epoch, cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr * config_dict["backbone_lr_ratio"]
        optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr

        for step, (image, mask, edge) in enumerate(loader):
            image, mask = image.cuda().float(), mask.cuda().float()

            out= net(image)
            loss = total_loss(out, mask)

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
            optimizer.step()

            ## log
            global_step += 1
            sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'loss': loss.item()}, global_step=global_step)
            if step % 10 == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | loss=%.6f' % (datetime.datetime.now(), global_step, epoch+1,
                                                cfg.epoch, optimizer.param_groups[0]['lr'], loss.item()))

        if config_dict["val"]:
            if epoch > cfg.epoch*config_dict["val_ratio"]-1:
                mae = validate(net, val_loader)
                print('ECSSD MAE:%s' % mae)
                if mae < min_mae:
                    min_mae = mae
                    best_epoch = epoch + 1
                    torch.save(net.state_dict(), cfg.savepath + '/model-' + str(epoch + 1))
                print('best epoch is:%d, MAE:%s' % (best_epoch, min_mae))
        if epoch >= cfg.epoch-1-config_dict["save_epoch_num"]:
            torch.save(net.state_dict(), cfg.savepath + '/model-' + str(epoch + 1))
    del net
    if config_dict["val"]:
        return best_epoch
    else:
        return 1




class Test(object):
    def __init__(self, Dataset, Network,batch_size, path, model, save_path, Resize):
        ## dataset
        self.save_path = save_path
        self.model  = model
        self.cfg    = Dataset.Config(datapath=path, snapshot=model, mode='test' ,resize=Resize )
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=batch_size, shuffle=False)
        ## network
        self.net    = Network()
        
        self.net.load_state_dict(torch.load(model))
        self.net.cuda()
        self.net.train(False)


    def save(self):
        head  = self.save_path +self.model.split('/')[-1] +'/'+ self.cfg.datapath.split('/')[-1]
        if not os.path.exists(head):
            os.makedirs(head)
        with torch.no_grad():
            for image, mask, shape, name in self.loader:
                # torch.cuda.synchronize()
                image = image.cuda().float()
                out = self.net(image)
                # print(shape)
                # out = F.interpolate(out, size=shape, mode='bilinear', align_corners=True)
                # pred = torch.sigmoid(out).cpu().numpy() * 255
                for i, once in enumerate(out):
                    once = torch.unsqueeze(once, dim=0) 
                    once = F.interpolate(once, size=(shape[0][i].item(),shape[1][i].item()), mode='bilinear', align_corners=True)
                    once = torch.sigmoid(once).cpu().numpy() * 255
                    cv2.imwrite(head+'/'+name[i]+'.png', once[0][0])


def test_dataset(method, config_dict):
    '''

    '''
    # 开始推断
    if config_dict["start_test"]:

        # 遍历每个模型
        for model in tqdm(config_dict["model_name"]):
            # 开始预测, 循环每一个数据集
            for path in tqdm(config_dict["data_path"] + x for x in config_dict["data_name"]):

                t = Test(dataset, 
                        mynet,batch_size=config_dict["batch"], path=path, model = config_dict["model_path"]+model, 
                        save_path = config_dict["save_path"] + method + '/',
                        Resize = config_dict["resize"])

                t.save()
    
    if config_dict["start_eval"]:
        # 开始评估
        # 计算文件head
        head = "model_name,remark,"
        for i in config_dict["data_name"]:
            for j in config_dict["Evaluation_indicators"]:
                head += j + "_" + i + ","
        head += "\n"

        
        if os.path.exists(config_dict["save_eval_csv"]):
            # 如果存在: 对比头
            file=open(config_dict["save_eval_csv"], "r+")
            if file.readline() != head: # 头不等就追加新的头
                file.seek(0,2)
                file.write(head)
            file.close()
        else:
            # 不存在,新建一个
            file=open(config_dict["save_eval_csv"], "a")
            file.close()
            file=open(config_dict["save_eval_csv"], "r+")
            file.write(head)
            file.close()
    # 循环每个模型
    for model in config_dict["model_name"]:
        # 循环每个数据集
        result_once = method + "," + model + ","
        for data_name in config_dict["data_name"]:
            pred_root = '{}{}/{}/{}/'.format(config_dict["save_path"], method, model, data_name)


            mask_root = '{}{}/mask'.format(config_dict["data_path"], data_name)
            mask_name_list = sorted(os.listdir(mask_root))
            FM = Fmeasure()
            WFM = WeightedFmeasure()
            SM = Smeasure()
            EM = Emeasure()
            M = MAE()
            for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
                mask_path = os.path.join(mask_root, mask_name)
                pred_path = os.path.join(pred_root, mask_name)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                FM.step(pred=pred, gt=mask)
                WFM.step(pred=pred, gt=mask)
                SM.step(pred=pred, gt=mask)
                EM.step(pred=pred, gt=mask)
                M.step(pred=pred, gt=mask)

            fm = FM.get_results()["fm"]
            wfm = WFM.get_results()["wfm"]
            sm = SM.get_results()["sm"]
            em = EM.get_results()["em"]
            mae = M.get_results()["mae"]
            result_once += str(round(sm,config_dict["Decimal_places"])) + ","
            result_once += str(round(mae,config_dict["Decimal_places"])) + ","
            result_once += str(round(wfm,config_dict["Decimal_places"])) + ","
            result_once += str(round(fm["curve"].max(),config_dict["Decimal_places"])) + ","
            result_once += str(round(fm["curve"].mean(),config_dict["Decimal_places"])) + ","
            result_once += str(round(fm["adp"],config_dict["Decimal_places"])) + ","
            result_once += str(round(em["curve"].max(),config_dict["Decimal_places"])) + ","
            result_once += str(round(em["curve"].mean(),config_dict["Decimal_places"])) + ","
            result_once += str(round(em["adp"],config_dict["Decimal_places"])) + ","
        file=open(config_dict["save_eval_csv"], "r+")
        file.seek(0,2)
        file.write(result_once+"\n")
        file.close()





if __name__ == '__main__':
    sys.path.insert(0, '../')
    sys.dont_write_bytecode = True
    method= os.getcwd().split("/")[-1] # 获取文件夹名字为method
    import glob
    import shutil
    
    # load config file
    with open('config.yaml',encoding="utf-8") as f:
        config_dict = yaml.load(f, Loader=SafeLoader)

    if config_dict["model_name"] == False:
        config_dict["model_name"] = []
    os.environ["CUDA_VISIBLE_DEVICES"] = config_dict["CUDA_DEVICE"]
    

    if config_dict["start_train"]:
        # for file in glob.glob("./out/events.*"):
        #     os.remove(file)
        # try:
        #     shutil.rmtree("./out/loss")
        # except:
        #     pass
        if config_dict["val"]:
            best_epoch = train(dataset, mynet, config_dict)
            best_model = "model-" + str(best_epoch)
            if config_dict["start_test"] and best_model not in config_dict["model_name"]:
                config_dict["model_name"].append(best_model)
        else:
            train(dataset, mynet, config_dict)
            print("Train Done!")
    if config_dict["method_name"] != False:
        method = config_dict["method_name"]
    test_dataset(method, config_dict)
    print("Done!!!")



    

