import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy
import json
import argparse
import datetime
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

EPS = 1e-12
rescale = 0.021377353 # 0.023123603


def mse(pred, label):
    loss = (pred - label)**2
    return torch.mean(loss)


def mae(pred, label):
    loss = (pred - label).abs()
    return torch.mean(loss)
    
def cal_cos_similarity(x, y): # the 2nd dimension of x and y are the same
    xy = x.mm(torch.t(y))
    x_norm = torch.sqrt(torch.sum(x*x, dim =1)).reshape(-1, 1)
    y_norm = torch.sqrt(torch.sum(y*y, dim =1)).reshape(-1, 1)
    cos_similarity = xy/x_norm.mm(torch.t(y_norm))
    cos_similarity[cos_similarity != cos_similarity] = 0
    return cos_similarity

class DataLoader:

    def __init__(self, df_feature, df_label, df_market_value, df_label0, batch_size=800, pin_memory=False, start_index = 0):

        assert len(df_feature) == len(df_label)

        self.df_feature = df_feature.values
        self.df_label = df_label.values
        self.df_market_value = df_market_value
        self.df_label0 = df_label0

        if pin_memory:
            self.df_feature = torch.tensor(self.df_feature, dtype=torch.float, device=device)
            self.df_label = torch.tensor(self.df_label, dtype=torch.float, device=device)
            self.df_market_value = torch.tensor(self.df_market_value, dtype=torch.float, device=device)
            self.df_label0 = torch.tensor(self.df_label0, dtype=torch.float, device=device)

        self.index = df_label.index

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.start_index = start_index

        self.daily_count = df_label.groupby(level=0).size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0

    @property
    def batch_length(self):

        if self.batch_size <= 0:
            return self.daily_length

        return len(self.df_label) // self.batch_size

    @property
    def daily_length(self):

        return len(self.daily_count)

    def iter_batch(self):

        if self.batch_size <= 0:
            yield from self.iter_daily_shuffle()
            return

        indices = np.arange(len(self.df_label))
        np.random.shuffle(indices)

        for i in range(len(indices))[::self.batch_size]:
            if len(indices) - i < self.batch_size:
                break
            yield indices[i:i+self.batch_size] # NOTE: advanced indexing will cause copy

    def iter_daily_shuffle(self):
        indices = np.arange(len(self.daily_count))
        np.random.shuffle(indices)
        for i in indices:
            yield i, slice(self.daily_index[i], self.daily_index[i] + self.daily_count[i])

    def iter_daily(self):
        indices = np.arange(len(self.daily_count))
        for i in indices:
            yield i, slice(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        # for idx, count in zip(self.daily_index, self.daily_count):
        #     yield slice(idx, idx + count) # NOTE: slice index will not cause copy

    def get(self, slc):
        outs = self.df_feature[slc], self.df_label[slc], self.df_market_value[slc], self.df_label0[slc]

        if not self.pin_memory:
            outs = tuple(torch.tensor(x, dtype=torch.float, device=device) for x in outs)

        return outs + (self.index[slc],)


class GRU(nn.Module):

    def __init__(self, d_feat=6, hidden_size=128, num_layers=2, dropout=0.0):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)

        self.d_feat = d_feat

    def forward(self, x):
        # x: [N, F*T]
        x = x.reshape(len(x), self.d_feat, -1) # [N, F, T]
        x = x.permute(0, 2, 1) # [N, T, F]
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()

class LSTM(nn.Module):

    def __init__(self, d_feat=6, hidden_size=128, num_layers=2, dropout=0.0):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)

        self.d_feat = d_feat

    def forward(self, x):
        # x: [N, F*T]
        x = x.reshape(len(x), self.d_feat, -1) # [N, F, T]
        x = x.permute(0, 2, 1) # [N, T, F]
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()


class MLP(nn.Module):

    def __init__(self, d_feat, hidden_size=512, num_layers=3, dropout=0.0):
        super().__init__()

        self.mlp = nn.Sequential()

        for i in range(num_layers):
            if i > 0:
                self.mlp.add_module('drop_%d'%i, nn.Dropout(dropout))
            self.mlp.add_module('fc_%d'%i, nn.Linear(
                180 if i == 0 else hidden_size, hidden_size))
            self.mlp.add_module('relu_%d'%i, nn.ReLU())

        self.mlp.add_module('fc_out', nn.Linear(hidden_size, 1))

    def forward(self, x):
        # feature
        # [N, F]
        return self.mlp(x).squeeze()


class HIST(nn.Module):

    def __init__(self, d_feat=6, hidden_size=128, num_layers=2, dropout=0.0, num_feature=180):
        super().__init__()

        self.d_feat = d_feat
        self.num_feature = num_feature
        self.hidden_size = hidden_size

        self.gru = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
                
        self.fc_es = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_es.weight)
        self.fc_is = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_is.weight)

        self.fc_es_middle = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_es_middle.weight)
        self.fc_is_middle = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_is_middle.weight)

        self.fc_es_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_es_fore.weight)
        self.fc_is_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_is_fore.weight)
        self.fc_indi_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_indi_fore.weight)

        self.fc_es_back = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_es_back.weight)
        self.fc_is_back = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_is_back.weight)
        self.fc_indi = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_indi.weight)

        self.leaky_relu = nn.LeakyReLU()
        self.softmax_s2t = torch.nn.Softmax(dim = 0)
        self.softmax_t2s = torch.nn.Softmax(dim = 1)
        
        self.fc_out_es = nn.Linear(hidden_size, 1)
        self.fc_out_is = nn.Linear(hidden_size, 1)
        self.fc_out_indi = nn.Linear(hidden_size, 1)
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x, tag_matrix, market_value):
        x_hidden = x.reshape(len(x), self.d_feat, -1) # [N, F, T]
        x_hidden = x_hidden.permute(0, 2, 1) # [N, T, F]
        x_hidden, _ = self.gru(x_hidden)
        x_hidden = x_hidden[:, -1, :]

        # Predefined Tag Module
        market_value_matrix = market_value.reshape(market_value.shape[0], 1).repeat(1, tag_matrix.shape[1])
        stock_to_tag = tag_matrix * market_value_matrix
        
        stock_to_tag_sum = torch.sum(stock_to_tag, 0).reshape(1, -1).repeat(stock_to_tag.shape[0], 1)
        stock_to_tag_sum = stock_to_tag_sum.mul(tag_matrix)

        stock_to_tag_sum = stock_to_tag_sum + (torch.ones(stock_to_tag.shape[0], stock_to_tag.shape[1]).to(device))
        stock_to_tag = stock_to_tag / stock_to_tag_sum
        hidden = torch.t(stock_to_tag).mm(x_hidden)
        
        hidden = hidden[hidden.sum(1)!=0]
        stock_to_tag = x_hidden.mm(torch.t(hidden))
        # stock_to_tag = cal_cos_similarity(x_hidden, hidden)
        stock_to_tag = self.softmax_s2t(stock_to_tag)
        hidden = torch.t(stock_to_tag).mm(x_hidden)
        
        tag_to_stock = cal_cos_similarity(x_hidden, hidden) 
        tag_to_stock = self.softmax_t2s(tag_to_stock)

        e_shared_info = tag_to_stock.mm(hidden)
        e_shared_info = self.fc_es(e_shared_info)

        e_shared_back = self.fc_es_back(e_shared_info)
        output_es = self.fc_es_fore(e_shared_info)
        output_es = self.leaky_relu(output_es)

        pred_es = self.fc_out_es(output_es).squeeze()
        
        # Hidden Shared Information Module
        i_shared_info = x_hidden - e_shared_back
        hidden = i_shared_info
        i_stock_to_tag = cal_cos_similarity(i_shared_info, hidden)

        dim = i_stock_to_tag.shape[0]
        diag = i_stock_to_tag.diagonal(0)
        i_stock_to_tag = i_stock_to_tag * (torch.ones(dim, dim) - torch.eye(dim)).to(device)
        row = torch.linspace(0,dim-1,dim).to(device).long()
        column =i_stock_to_tag.max(1)[1].long()
        value = i_stock_to_tag.max(1)[0]
        i_stock_to_tag[row, column] = 10
        i_stock_to_tag[i_stock_to_tag!=10]=0
        i_stock_to_tag[row, column] = value
        i_stock_to_tag = i_stock_to_tag + torch.diag_embed((i_stock_to_tag.sum(0)!=0).float()*diag)
        hidden = torch.t(i_shared_info).mm(i_stock_to_tag).t()
        hidden = hidden[hidden.sum(1)!=0]

        i_tag_to_stock = cal_cos_similarity(i_shared_info, hidden)
        i_tag_to_stock = self.softmax_t2s(i_tag_to_stock)
        i_shared_info = i_tag_to_stock.mm(hidden)
        i_shared_info = self.fc_is(i_shared_info)

        i_shared_back = self.fc_is_back(i_shared_info)
        output_is = self.fc_is_fore(i_shared_info)
        output_is = self.leaky_relu(output_is)
        pred_is = self.fc_out_is(output_is).squeeze()

        # Individual Information Module
        individual_info  = x_hidden - e_shared_back - i_shared_back
        output_indi = individual_info
        output_indi = self.fc_indi(output_indi)
        output_indi = self.leaky_relu(output_indi)
        pred_indi = self.fc_out_indi(output_indi).squeeze()

        # Stock Trend Prediction
        all_info = output_es + output_is + output_indi
        pred_all = self.fc_out(all_info).squeeze()

        return pred_all


def get_model(model_name):

    if model_name.upper() == 'LSTM':
        return LSTM

    if model_name.upper() == 'GRU':
        return GRU

    if model_name.upper() == 'MLP':
        return MLP

    if model_name.upper() == 'HIST':
        return HIST

    raise ValueError('unknown model name `%s`'%model_name)


def average_params(params_list):
    assert isinstance(params_list, (tuple, list, collections.deque))
    n = len(params_list)
    if n == 1:
        return params_list[0]
    new_params = collections.OrderedDict()
    keys = None
    for i, params in enumerate(params_list):
        if keys is None:
            keys = params.keys()
        for k, v in params.items():
            if k not in keys:
                raise ValueError('the %d-th model has different params'%i)
            if k not in new_params:
                new_params[k] = v / n
            else:
                new_params[k] += v / n
    return new_params



def metric_fn(preds):
    preds = preds[~np.isnan(preds['label'])]

    ic = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x.score)).mean()
    rank_ic = preds.groupby(level='datetime').apply(lambda x: x.label.corr(x.score, method='spearman')).mean()

    preds = preds[~np.isnan(preds['label0'])]
    mae_score = preds['score'] * rescale / np.mean(abs(preds['score']))
    mae_label = preds['label'] * rescale / np.mean(abs(preds['label']))
    mse = np.mean((mae_label - mae_score)**2)
    mae = np.mean(abs(mae_label - mae_score))

    return mse, mae, ic, rank_ic



def loss_fn(pred, label, args):
    mask = ~torch.isnan(label)

    return mse(pred[mask], label[mask])


global_log_file = None
def pprint(*args):
    # print with UTC+8 time
    time = '['+str(datetime.datetime.utcnow()+
                   datetime.timedelta(hours=8))[:19]+'] -'
    print(time, *args, flush=True)

    if global_log_file is None:
        return
    with open(global_log_file, 'a') as f:
        print(time, *args, flush=True, file=f)


global_step = -1
def train_epoch(epoch, model, optimizer, train_loader, writer, tag_matrix, date_index, args):

    global global_step

    model.train()

    for i, slc in tqdm(train_loader.iter_batch(), total=train_loader.batch_length):

        global_step += 1

        feature, label, market_value, _, _ = train_loader.get(slc)
        if args.model_name == 'HIST':
            pred = model(feature, tag_matrix[date_index[i]], market_value)
        else:
            pred = model(feature)
        loss = loss_fn(pred, label, args)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
        optimizer.step()


def test_epoch(epoch, model, test_loader, writer, args, tag_matrix, date_index, prefix='Test'):

    model.eval()

    losses = []
    preds = []

    for i, slc in tqdm(test_loader.iter_daily(), desc=prefix, total=test_loader.daily_length):

        feature, label, market_value, label0, index = test_loader.get(slc)

        with torch.no_grad():
            if args.model_name == 'HIST':
                pred = model(feature, tag_matrix[date_index[test_loader.start_index + i]], market_value)
            else:
                pred = model(feature)
            loss = loss_fn(pred, label, args)
            preds.append(pd.DataFrame({ 'score': pred.cpu().numpy(), 'label': label.cpu().numpy(), 'label0': label0.cpu().numpy(), }, index=index))

        losses.append(loss.item())
    #evaluate
    preds = pd.concat(preds, axis=0)
    mse, mae, ic, rank_ic = metric_fn(preds)
    # scores = rank_ic
    scores = -1.0 * mae

    writer.add_scalar(prefix+'/Loss', np.mean(losses), epoch)
    writer.add_scalar(prefix+'/std(Loss)', np.std(losses), epoch)
    writer.add_scalar(prefix+'/'+args.metric, np.mean(scores), epoch)
    writer.add_scalar(prefix+'/std('+args.metric+')', np.std(scores), epoch)

    return np.mean(losses), scores, mse, mae, ic, rank_ic

def inference(model, data_loader, tag_matrix, date_index):

    model.eval()

    preds = []
    for i, slc in tqdm(data_loader.iter_daily(), total=data_loader.daily_length):

        feature, label, market_value, label0, index = data_loader.get(slc)
        with torch.no_grad():
            if args.model_name == 'HIST':
                pred = model(feature, tag_matrix[date_index[data_loader.start_index + i]], market_value)
            else:
                pred = model(feature)
            preds.append(pd.DataFrame({ 'score': pred.cpu().numpy(), 'label': label.cpu().numpy(), 'label0': label0.cpu().numpy(), }, index=index))

    preds = pd.concat(preds, axis=0)
    return preds


def create_loaders(args):

    df = pd.read_pickle(args.data_path)
    # NOTE: we always assume the last column is label
    df_feature = df.iloc[:, :-4]
    df_label = df.iloc[:, -2]
    df_label0 = df.iloc[:, -3]
    df_market_value = df.iloc[:, -1]/1000000000
    df_market_value.fillna(0, inplace=True)

    start_index = 0
    slc = slice(pd.Timestamp(args.train_start_date), pd.Timestamp(args.train_end_date))

    mean = df_feature.loc[slc].mean()
    std = df_feature.loc[slc].std()
    df_feature = (df_feature - mean) /std
    df_feature.fillna(0, inplace=True)
    train_label0 = df_label0[slc]
    train_label0 = train_label0[~np.isnan(train_label0)]
    global rescale
    rescale = np.mean(abs(train_label0))
    train_loader = DataLoader(df_feature.loc[slc], df_label.loc[slc], df_market_value[slc], df_label0[slc], batch_size=args.batch_size, pin_memory=args.pin_memory, start_index=start_index)

    start_index += len(df_label.loc[slc].groupby(level=0).size())
    slc = slice(pd.Timestamp(args.valid_start_date), pd.Timestamp(args.valid_end_date))
    valid_loader = DataLoader(df_feature.loc[slc], df_label.loc[slc], df_market_value[slc], df_label0[slc], pin_memory=False, start_index=start_index)

    start_index += len(df_label.loc[slc].groupby(level=0).size())
    slc = slice(pd.Timestamp(args.test_start_date), pd.Timestamp(args.test_end_date))
    test_loader = DataLoader(df_feature.loc[slc], df_label.loc[slc], df_market_value[slc], df_label0[slc], pin_memory=False, start_index=start_index)

    return train_loader, valid_loader, test_loader


def main(args):
    seed = np.random.randint(1000000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_trace()
    suffix = "%s_dh%s_dn%s_drop%s_lr%s_bs%s_seed%s%s"%(
        args.model_name, args.hidden_size, args.num_layers, args.dropout,
        args.lr, args.batch_size, args.seed, args.annot
    )

    output_path = args.outdir
    if not output_path:
        output_path = './output/' + suffix
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not args.overwrite and os.path.exists(output_path+'/'+'info.json'):
        print('already runned, exit.')
        return

    writer = SummaryWriter(log_dir=output_path)

    global global_log_file
    global_log_file = output_path + '/' + 'run.log'

    pprint('create model...')
    model = get_model(args.model_name)(args.d_feat, args.hidden_size, args.num_layers, args.dropout)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    pprint('create loaders...')
    train_loader, valid_loader, test_loader = create_loaders(args)

    tag_matrix = np.load(args.tag_matrix)
    date_index = np.load(args.date_index).astype(np.int32)
    if args.model_name == 'HIST':
        tag_matrix = torch.Tensor(tag_matrix).to(device)

    best_score = -np.inf
    best_epoch = 0
    stop_round = 0
    best_param = copy.deepcopy(model.state_dict())
    params_list = collections.deque(maxlen=args.smooth_steps)
    for epoch in range(args.n_epochs):
        pprint('Epoch:', epoch)

        pprint('training...')
        train_epoch(epoch, model, optimizer, train_loader, writer, tag_matrix, date_index, args)
        torch.save(model.state_dict(), output_path+'/model.bin.e'+str(epoch))
        torch.save(optimizer.state_dict(), output_path+'/optimizer.bin.e'+str(epoch))

        params_ckpt = copy.deepcopy(model.state_dict())
        params_list.append(params_ckpt)
        avg_params = average_params(params_list)
        model.load_state_dict(avg_params)

        pprint('evaluating...')
        train_loss, train_score, train_mse, train_mae, train_ic, train_rank_ic = test_epoch(epoch, model, train_loader, writer, args, tag_matrix, date_index, prefix='Train')
        val_loss, val_score, val_mse, val_mae, val_ic, val_rank_ic = test_epoch(epoch, model, valid_loader, writer, args, tag_matrix, date_index, prefix='Valid')
        test_loss, test_score, test_mse, test_mae, test_ic, test_rank_ic = test_epoch(epoch, model, test_loader, writer, args, tag_matrix, date_index, prefix='Test')

        pprint('train_loss %.6f, valid_loss %.6f, test_loss %.6f'%(train_loss, val_loss, test_loss))
        pprint('train_score %.6f, valid_score %.6f, test_score %.6f'%(train_score, val_score, test_score))
        pprint('train_mse %.6f, valid_mse %.6f, test_mse %.6f'%(train_mse, val_mse, test_mse))
        pprint('train_mae %.6f, valid_mae %.6f, test_mae %.6f'%(train_mae, val_mae, test_mae))
        pprint('train_ic %.6f, valid_ic %.6f, test_ic %.6f'%(train_ic, val_ic, test_ic))
        pprint('train_rank_ic %.6f, valid_rank_ic %.6f, test_rank_ic %.6f'%(train_rank_ic, val_rank_ic, test_rank_ic))

        model.load_state_dict(params_ckpt)

        if val_score > best_score:
            best_score = val_score
            stop_round = 0
            best_epoch = epoch
            best_param = copy.deepcopy(avg_params)
        else:
            stop_round += 1
            if stop_round >= args.early_stop:
                pprint('early stop')
                break

    pprint('best score:', best_score, '@', best_epoch)
    model.load_state_dict(best_param)
    torch.save(best_param, output_path+'/model.bin')

    pprint('inference...')
    res = dict()
    for name in ['train', 'valid', 'test']:

        pred= inference(model, eval(name+'_loader'), tag_matrix, date_index)
        pred.to_pickle(output_path+'/pred.pkl.'+name)

        ic = pred.groupby(level='datetime').apply(
            lambda x: x.label.corr(x.score))
        rank_ic = pred.groupby(level='datetime').apply(
            lambda x: x.label.corr(x.score, method='spearman'))
        preds = pred[~np.isnan(pred['label0'])]
        mae_score = preds['score'] * rescale / np.mean(abs(preds['score']))
        mae_label = preds['label'] * rescale / np.mean(abs(preds['label']))
        mse = np.mean((mae_label - mae_score)**2)
        mae = np.mean(abs(mae_label - mae_score))
        pprint(('%s: MSE %.6f, MAE %.6f, IC %.6f Rank IC %.6f')%(
                    name, mse, mae, ic.mean(), rank_ic.mean()))

        res[name+'-IC'] = ic.mean()
        res[name+'-ICIR'] = ic.mean() / ic.std()
        res[name+'-RankIC'] = rank_ic.mean()
        res[name+'-RankICIR'] = rank_ic.mean() / rank_ic.std()

    pprint('save info...')
    writer.add_hparams(
        vars(args),
        {
            'hparam/'+key: value
            for key, value in res.items()
        }
    )

    info = dict(
        config=vars(args),
        best_epoch=best_epoch,
        best_score=res,
    )
    default = lambda x: str(x)[:10] if isinstance(x, pd.Timestamp) else x
    with open(output_path+'/info.json', 'w') as f:
        json.dump(info, f, default=default, indent=4)

    pprint('finished.')


class ParseConfigFile(argparse.Action):

    def __call__(self, parser, namespace, filename, option_string=None):

        if not os.path.exists(filename):
            raise ValueError('cannot find config at `%s`'%filename)

        with open(filename) as f:
            config = json.load(f)
            for key, value in config.items():
                setattr(namespace, key, value)


def parse_args():

    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_name', default='HIST')
    parser.add_argument('--d_feat', type=int, default=6)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)

    # training
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--early_stop', type=int, default=30)
    parser.add_argument('--smooth_steps', type=int, default=5)
    parser.add_argument('--metric', default='IC')
    parser.add_argument('--loss', default='mse')

    # data
    parser.add_argument('--pin_memory', action='store_false', default=True)
    parser.add_argument('--batch_size', type=int, default=-1) # -1 indicate daily batch
    parser.add_argument('--least_samples_num', type=float, default=1137.0) 
    parser.add_argument('--label', default='') # specify other labels
    parser.add_argument('--train_start_date', default='2007-01-01')
    parser.add_argument('--train_end_date', default='2014-12-31')
    parser.add_argument('--valid_start_date', default='2015-01-01')
    parser.add_argument('--valid_end_date', default='2016-12-31')
    parser.add_argument('--test_start_date', default='2017-01-01')
    parser.add_argument('--test_end_date', default='2019-12-31')

    # other
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--annot', default='')
    parser.add_argument('--config', action=ParseConfigFile, default='')

    # input for csi 300
    parser.add_argument('--data_path', default='./data/csi300_07to19_30days.pkl')
    parser.add_argument('--tag_matrix', default='./data/csi300_stock2tag_matrix.npy')
    parser.add_argument('--date_index', default='./data/csi300_date_index.npy')
    
    # For csi 500 dataset, use the following inputs.
    # parser.add_argument('--data_path', default='./data/csi500_07to19_30days.pkl')
    # parser.add_argument('--tag_matrix', default='./data/csi500_stock2tag_matrix.npy')
    # parser.add_argument('--date_index', default='./data/csi500_date_index.npy')

    parser.add_argument('--outdir', default='./output/csi300_hist')
    parser.add_argument('--overwrite', action='store_true', default=False)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    main(args)