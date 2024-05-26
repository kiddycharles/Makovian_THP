import os
import torch
from Informer.exp.exp_informer import Exp_Informer
from Informer.data.data_loader import Dataset_Pred
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

Exp = Exp_Informer


class InformerConfig:
    def __init__(self):
        self.model = 'informer'
        self.data = '^IXIC'
        self.root_path = './equities/stocks/'
        self.data_path = '^IXIC.csv'
        self.features = 'M'
        self.target = 'Adj Close'
        self.freq = 'b'
        self.checkpoints = './checkpoints/'
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 24
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        self.d_model = 512
        self.n_heads = 8
        self.e_layers = 2
        self.d_layers = 1
        self.s_layers = '3,2,1'
        self.d_ff = 2048
        self.factor = 5
        self.padding = 0
        self.distil = True
        self.dropout = 0.05
        self.attn = 'prob'
        self.embed = 'timeF'
        self.activation = 'gelu'
        self.output_attention = False
        self.do_predict = False
        self.mix = True
        self.cols = None
        self.num_workers = 0
        self.itr = 2
        self.train_epochs = 6
        self.batch_size = 32
        self.patience = 3
        self.learning_rate = 0.0001
        self.des = 'test'
        self.loss = 'mse'
        self.lradj = 'type1'
        self.use_amp = False
        self.inverse = False
        self.use_gpu = True
        self.gpu = 0
        self.use_multi_gpu = False
        self.devices = '0,1,2,3'


args = InformerConfig()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'WTH': {'data': 'WTH.csv', 'T': 'WetBulbCelsius', 'M': [12, 12, 12], 'S': [1, 1, 1], 'MS': [12, 12, 1]},
    'ECL': {'data': 'ECL.csv', 'T': 'MT_320', 'M': [321, 321, 321], 'S': [1, 1, 1], 'MS': [321, 321, 1]},
    'Solar': {'data': 'solar_AL.csv', 'T': 'POWER_136', 'M': [137, 137, 137], 'S': [1, 1, 1], 'MS': [137, 137, 1]},
    '^IXIC': {'data': '^IXIC.csv', 'T': 'Adj Close', 'M': [6, 6, 6], 'S': [1, 1, 1], 'MS': [6, 6, 1]}
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ', '').split(',')]
args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)

Data = Dataset_Pred
timeenc = 0 if args.embed!='timeF' else 1
flag = 'pred'; shuffle_flag = False; drop_last = False; batch_size = 1

freq = args.detail_freq

data_set = Data(
    root_path=args.root_path,
    data_path=args.data_path,
    flag=flag,
    size=[args.seq_len, args.label_len, args.pred_len],
    features=args.features,
    target=args.target,
    timeenc=timeenc,
    freq=freq
)
data_loader = DataLoader(
    data_set,
    batch_size=batch_size,
    shuffle=shuffle_flag,
    num_workers=args.num_workers,
    drop_last=drop_last)

args.output_attention = True

exp = Exp(args)

model = exp.model

setting = "informer_^IXIC_ftM_sl96_ll96_pl48_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_'Exp'_0"
path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
model.load_state_dict(torch.load(path))

# attention visualization
idx = 0
for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
    if i != idx:
        continue
    batch_x = batch_x.float().to(exp.device)
    batch_y = batch_y.float()

    batch_x_mark = batch_x_mark.float().to(exp.device)
    batch_y_mark = batch_y_mark.float().to(exp.device)

    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(exp.device)

    outputs, attn = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    attn[0].shape, attn[1].shape #, attn[2].shape
    print(attn[0].shape, attn[1].shape)
    layer = 0
    distil = 'Distil' if args.distil else 'NoDistil'
    for h in range(0,8):
        plt.figure(figsize=[10,8])
        plt.title('Informer, {}, attn:{} layer:{} head:{}'.format(distil, args.attn, layer, h))
        A = attn[layer][0,h].detach().cpu().numpy()
        ax = sns.heatmap(A, vmin=0, vmax=A.max()+0.01)
        plt.show()


    layer = 1
    distil = 'Distil' if args.distil else 'NoDistil'
    for h in range(0,8):
        plt.figure(figsize=[10,8])
        plt.title('Informer, {}, attn:{} layer:{} head:{}'.format(distil, args.attn, layer, h))
        A = attn[layer][0,h].detach().cpu().numpy()
        ax = sns.heatmap(A, vmin=0, vmax=A.max()+0.01)
        plt.show()

print(len(data_set), len(data_loader))
# When we finished exp.train(setting) and exp.test(setting), we will get a trained model and the results of test experiment
# The results of test experiment will be saved in ./results/{setting}/pred.npy (prediction of test dataset) and ./results/{setting}/true.npy (groundtruth of test dataset)

preds = np.load('./results/'+setting+'/pred.npy')
trues = np.load('./results/'+setting+'/true.npy')

# [samples, pred_len, dimensions]
print(preds.shape, trues.shape)
# draw OT prediction
plt.figure()
plt.plot(trues[0,:,-1], label='GroundTruth')
plt.plot(preds[0,:,-1], label='Prediction')
plt.legend()
plt.show()

# draw HUFL prediction
plt.figure()
plt.plot(trues[0,:,0], label='GroundTruth')
plt.plot(preds[0,:,0], label='Prediction')
plt.legend()
plt.show()

