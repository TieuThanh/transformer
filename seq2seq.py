from processing import *
from transformer import Transformer
from torch.nn.functional import nll_loss

BATCH_SIZE = 16

X_train, Y_train = readData('./data/train.en','./data/train.vi')
X_test, Y_test = readData('./data/tst2013.en','./data/tst2013.vi')
test_data,_,_ = processingData(X_test,Y_test,'vi','vi')
train_data, source_filed, target_filed = processingData(X_train,Y_train,'vi','vi')
source_padding = source_filed.vocab.stoi['<pad>']
target_padding = source_filed.vocab.stoi['<pad>']

train_iter = Iterator(train_data,batch_size=BATCH_SIZE, sort_within_batch=False, repeat=False, sort_key=lambda x: (len(source_filed.vocab),len(target_filed.vocab)))
test_iter = Iterator(test_data,batch_size=BATCH_SIZE, sort_within_batch=False, repeat=False, sort_key=lambda x: (len(source_filed.vocab),len(target_filed.vocab)))
if len (source_filed.vocab) > len(target_filed.vocab):
    model = Transformer(len(source_filed.vocab))
else:
    model = Transformer(len(target_filed.vocab))

optimizer = torch.optim.Adam(model.parameters(),lr = 0.0001,betas=(0.9,0.98),eps=1e-09)

for p in model.parameters():
    if p.dim() >1:
        torch.nn.init.xavier_uniform_(p)
EPOCHS = 10

for epoch in range(EPOCHS):
    dem = 0
    loss_sum = 0
    model.train()
    for batch in train_iter:
        b_input = batch.source.transpose(0,1)
        b_target = batch.target.transpose(0,1)
        b_input_size = b_input.size(-1)
        b_target_size = b_target.size(-1)
        if b_input_size < b_target_size:
            temp = torch.zeros((BATCH_SIZE,b_target_size))
            temp[:,:b_input_size] = b_input
            temp[:,b_input_size:] = 1
            b_input = temp.long()
        elif b_input_size > b_target_size:
            temp = torch.zeros((BATCH_SIZE,b_input_size))
            temp[:,:b_target_size] = b_target
            temp[:,b_target_size:] = 1
            b_target = temp.long()
        src_mask, tgt_mask = createMasks(b_input,b_target,source_padding,target_padding)
        preds = model(b_input,b_target)
        # print(preds.shape)
        # print(b_target.shape)
        optimizer.zero_grad()
        loss = nll_loss(preds.transpose(2,1),b_target)
        loss.backward()
        optimizer.step()
        # print("loss: {}".format(loss.item()))
        dem+=1
        loss_sum += loss.item()
    print("Epoch = {}, loss = {}".format(epoch + 1, loss_sum/dem))
        
