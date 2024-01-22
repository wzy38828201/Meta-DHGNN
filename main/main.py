import torch
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
from utils import load_data, EarlyStopping
import random
import warnings
warnings.filterwarnings("ignore")

# 禁用gpu
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"

random.seed(20)
np.random.seed(20)
torch.manual_seed(20)
torch.cuda.manual_seed(20)
torch.cuda.manual_seed_all(20)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def Combinations(L, k):
    """List all combinations: choose k elements from list L"""
    na = len(L)
    result = [] # To Place Combination result
    for i in range(na-k+1):
        if k > 1:
            newL = L[i+1:]
            Comb, _ = Combinations(newL, k - 1)
            for item in Comb:
                item.insert(0, L[i])
                result.append(item)
        else:
            result.append([L[i]])
    return result, len(result)

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')#
    macro_f1 = f1_score(labels, prediction, average='macro')
    cr = classification_report(labels, prediction, digits=4)
    cm = confusion_matrix(labels, prediction)#'ACM/labels.csv''ACM/prediction.csv','ACM/logits.csv'

    return accuracy, micro_f1, macro_f1,cr, logits, labels, prediction

def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1,cr, logitss, labelss, predictions = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1,cr,logitss, labelss, predictions

def evaluate1(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1,cr, logitss, labelss, predictions = score(logits[mask], labels[mask])

    return logitss, labelss, predictions

def main(args):
    File = open(r"\draw\Parameter.txt", 'a+')

    # If args['hetero'] is True, g would be a heterogeneous graph.
    # Otherwise, it will be a list of homogeneous graphs.
    g, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
    val_mask, test_mask = load_data(args['dataset'])

    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()

    features = features.to(args['device'])
    labels = labels.to(args['device'])
    train_mask = train_mask.to(args['device'])
    val_mask = val_mask.to(args['device'])
    test_mask = test_mask.to(args['device'])
    train_maskk = train_mask
    val_maskk = val_mask

    if args['hetero']:
        print('aaaaa')
        from model_hetero import HAN, Res_HAN
        model = Res_HAN(meta_paths=[['pa', 'ap'], ['pf', 'fp']],
                    in_size=features.shape[1],
                    hidden_size=args['hidden_units'],
                    out_size=num_classes,
                    num_heads=args['num_heads'],
                    dropout=args['dropout']).to(args['device'])
        g = g.to(args['device'])
    else:
        print('bbbbb')
        from model import HAN, Res_HAN
        model = Res_HAN(num_meta_paths=len(g),
                    in_size=features.shape[1],
                    hidden_size=args['hidden_units'],
                    out_size=num_classes,
                    num_heads=args['num_heads'],
                    dropout=args['dropout']).to(args['device'])
        g = [graph.to(args['device']) for graph in g]

    stopper = EarlyStopping(patience=args['patience'])
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],#'lr'
                                 weight_decay=args['weight_decay'])
    optimizer2 = torch.optim.Adam(model.parameters(), lr=args['lr2'],
                                 weight_decay=args['weight_decay2'])
     #——————Meta——————
    step = args['step']  # Depending on how many metagrams you want to generate

    train_shot_0 = args['train_shot_0']
    #val_shot_1 = 35

    select_class = [0, 1, 2]
    class1_idx = []
    class2_idx = []
    class3_idx = []
    train_idx_i = train_idx

    y = labels.tolist()

    r0 = []
    r1 = []
    r2 = []
    for uuu in y:
        if int(uuu)==0:
            r0.append(1)
            r1.append(0)
            r2.append(0)
        elif int(uuu)==1:
            r1.append(1)
            r0.append(0)
            r2.append(0)
        elif int(uuu)==2:
            r2.append(1)
            r0.append(0)
            r1.append(0)
    p = pd.DataFrame()
    p[0] = r0
    p[1] = r1
    p[2] = r2
    truelabels = np.array(p)

    n1_ = Combinations(select_class, 3)
    labels_local = pd.DataFrame()
    labels_local[0] = list(0 for r in range(len(truelabels)))
    labels_local[1] = list(0 for r in range(len(truelabels)))
    labels_local[2] = list(0 for r in range(len(truelabels)))
    labels_local = np.array(labels_local)
    for ru in train_idx:
        ru = int(ru)
        labels_local[ru] = truelabels[ru]
    for select_class_i in n1_[0]:
        se = select_class_i[0]
        se1 = select_class_i[1]
        se2 = select_class_i[2]
        for k in train_idx_i:
            k = int(k)
            if (int(pd.DataFrame(labels_local)[se][k]) == 1):
                class1_idx.append(k)
            elif (int(pd.DataFrame(labels_local)[se1][k]) == 1):
                class2_idx.append(k)
            elif (int(pd.DataFrame(labels_local)[se2][k]) == 1):
                class3_idx.append(k)
        for m in range(step):
            # if m!=0:
            #     stopper.load_checkpoint(model)
            # else:
            #     sdf = 2
            class1_train = random.sample(class1_idx, train_shot_0)
            class2_train = random.sample(class2_idx, train_shot_0)
            class3_train = random.sample(class3_idx, train_shot_0)
            class1_val = [n1 for n1 in class1_idx if n1 not in class1_train]
            class2_val = [n2 for n2 in class2_idx if n2 not in class2_train]
            class3_val = [n3 for n3 in class3_idx if n3 not in class3_train]

            train_idx2 = class1_train+class2_train+class3_train
            #random.shuffle(train_id3x2)
            val_idx2 = class1_val+class2_val+class3_val
            #random.shuffle(val_idx2)

            #y = labels_local
            train_idx2 = np.array(train_idx2)
            val_idx2 = np.array(val_idx2)

            # train2 = np.array([x for x in train_idx.tolist() if x not in range(len(train_idx2))])
            # val2 = np.array([x for x in val_idx.tolist() if x not in range(len(val_idx2))])

            train_mask = pd.DataFrame()
            train_mask[0] = [0 for r in range(len(truelabels))]
            train_mask = np.array(train_mask)
            for ru in train_idx2:
                ru = int(ru)
                train_mask[ru] = True
            train_mask = train_mask.reshape(1, len(train_mask)).tolist()[0]
            train_mask = torch.tensor(train_mask).bool()
            train_mask = train_mask.to(args['device'])
            #train_mask = pd.DataFrame(train_mask)

            val_mask = pd.DataFrame()
            val_mask[0] = [0 for r in range(len(truelabels))]
            val_mask = np.array(val_mask)
            for ru in val_idx2:
                ru = int(ru)
                val_mask[ru] = True
            val_mask = val_mask.reshape(1,len(val_mask)).tolist()[0]
            val_mask = torch.tensor(val_mask).bool()
            val_mask = val_mask.to(args['device'])
            li0 = {}
            for epoch in range(args['num_epochs']):
                model.train()
                logits = model(g, features)
                loss = loss_fcn(logits[train_mask], labels[train_mask])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_acc, train_micro_f1, train_macro_f1,_ ,logitsss, labelsss, predictionss= score(logits[train_mask], labels[train_mask])
                val_loss, val_acc, val_micro_f1, val_macro_f1,_ ,logi, labe, predictio= evaluate(model, g, features, labels, val_mask, loss_fcn)


                print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
                      'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
                    epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1, val_macro_f1))
                li0[epoch] = val_macro_f1
                torch.save(model,r'\save0\model_weights.pth' + str(epoch))

                # if early_stop:
                #     stopper.save_checkpoint(model)
                #     break
    max_key0 = max(li0.items(), key=lambda x: x[1])[0]
    print(max_key0,li0[max_key0])

    torch.save(model.state_dict(), r'\save0\model_weights.pth'+str(max_key0))
    print('aaaaaaaaaaaaaaaaaaaaaaa')
    li = {}
    for epoch1 in range(args['num_epochs1']):
        model.train()
        logits = model(g, features)
        loss2 = loss_fcn(logits[train_maskk], labels[train_maskk])

        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        train_acc, train_micro_f11, train_macro_f11,_, logitss, labelss, predictions= score(logits[train_maskk], labels[train_maskk])
        val_loss1, val_acc1, val_micro_f11, val_macro_f11,_ ,logitssg, labelssg, predictionsg= evaluate(model, g, features, labels, val_maskk, loss_fcn)
        #early_stop = stopper.step(val_loss1.data.item(), val_acc1, model)
        torch.save(model, r'\save\model_weights.pth'+str(epoch1))

        li[epoch1] =  val_macro_f11
        print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
              'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
            epoch1 + 1, loss2.item(), train_micro_f11, train_macro_f11, val_loss1.item(), val_micro_f11, val_macro_f11))
        # if early_stop:
        #     stopper.save_checkpoint(model)
        #     break

    max_key = max(li.items(), key=lambda x: x[1])[0]
    print(max_key,li[max_key])
    #stopper.save_checkpoint(model)

    #读取model
    model = torch.load(r'\save\model_weights.pth'+str(max_key), map_location=args['device'])
    #stopper.load_checkpoint(model)

    test_loss, test_acc, test_micro_f1, test_macro_f1,test_cr,logitssy, labelssy, predictionsy = evaluate(model, g, features, labels, test_mask, loss_fcn)
    print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
        test_loss.item(), test_micro_f1, test_macro_f1))
    print(test_cr)

    File.write(str(max_key)+"\t"+str(test_macro_f1)+'\n')
    File.flush()


    pd.DataFrame(labelssy).to_csv(r'\results\all\IMDB\labels.csv', index=False, header=False)
    pd.DataFrame(predictionsy).to_csv(r'\results\all\IMDB\prediction.csv', index=False, header=False)
    pd.DataFrame(logitssy).to_csv(r'\results\all\IMDB\logits.csv', index=False, header=False)


if __name__ == '__main__':
    import argparse
    import sys
    from utils import setup

    parser = argparse.ArgumentParser('HAN')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')
    parser.add_argument('--hetero', action='store_true',
                        help='Use metapath coalescing with DGL\'s own dataset')

    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--lr2', type=float, default=0.006)#
    parser.add_argument('--num_heads', type=list, default=[8,8])
    parser.add_argument('--hidden_units', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--weight_decay2', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--num_epochs1', type=int, default=150)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='IMDB')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--step', type=int, default=2)#ACM:2
    parser.add_argument('--train_shot_0', type=int, default=25)

    args = parser.parse_args().__dict__

    #args = setup(args)

    main(args)
