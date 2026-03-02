import sys
sys.path.append('/home/srwang/trend_forecasting/dataprovider')
from data_loader import Dataset_Finance, Dataset_Finance_new
from torch.utils.data import DataLoader
import pandas as pd

data_dict = {
    'index': Dataset_Finance_new,
}


def data_provider(args, flag, label_num):
    Data = data_dict[args.data]

    if flag == 'test' or flag=='val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    # elif flag == 'pred':
    #     shuffle_flag = False
    #     drop_last = False
    #     batch_size = 1
    #     freq = args.freq
    #     Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    if args.data == 'index':
        data_set = Data(
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.pred_len],
            data_name=args.data,
            stock_num=args.num_nodes,
            label_num=label_num
        )

    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
