import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import random
import math
import time
import os 
from torch.utils.data import TensorDataset, DataLoader

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class Encoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 emb_dim, 
                 hid_dim, 
                 n_layers, 
                 kernel_size, 
                 dropout, 
                 device,
                 seq_length = 100):
        super().__init__()
                
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        
        self.tok_embedding = nn.Embedding(input_dim, emb_dim)
        self.pos_embedding = nn.Embedding(seq_length, emb_dim)
        
        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)
        
        self.convs = nn.ModuleList([nn.Conv1d(in_channels = hid_dim, 
                                              out_channels = 2 * hid_dim, 
                                              kernel_size = kernel_size, 
                                              padding = (kernel_size - 1) // 2)
                                    for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
                
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        tok_embedded = self.tok_embedding(src)
        pos_embedded = self.pos_embedding(pos)

        embedded = self.dropout(tok_embedded + pos_embedded)

        conv_input = self.emb2hid(embedded)
        conv_input = conv_input.permute(0, 2, 1) 

        for i, conv in enumerate(self.convs):
        
            conved = conv(self.dropout(conv_input))
            conved = F.glu(conved, dim = 1)
            conved = (conved + conv_input) * self.scale
            conv_input = conved
        
        conved = self.hid2emb(conved.permute(0, 2, 1))
        combined = (conved + embedded) * self.scale
                
        return conved, combined

class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 emb_dim, 
                 hid_dim, 
                 n_layers, 
                 kernel_size, 
                 dropout, 
                 trg_pad_idx, 
                 device,
                 seq_length = 100):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        
        self.tok_embedding = nn.Embedding(output_dim, emb_dim)
        self.pos_embedding = nn.Embedding(seq_length, emb_dim)
        
        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)
        
        self.attn_hid2emb = nn.Linear(hid_dim, emb_dim)
        self.attn_emb2hid = nn.Linear(emb_dim, hid_dim)
        
        self.fc_out = nn.Linear(emb_dim, 1)
        
        self.convs = nn.ModuleList([nn.Conv1d(in_channels = hid_dim, 
                                              out_channels = 2 * hid_dim, 
                                              kernel_size = kernel_size)
                                    for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
      
    def calculate_attention(self, embedded, conved, encoder_conved, encoder_combined):
        
        conved_emb = self.attn_hid2emb(conved.permute(0, 2, 1))        
        combined = (conved_emb + embedded) * self.scale
                    
        energy = torch.matmul(combined, encoder_conved.permute(0, 2, 1))      
        attention = F.softmax(energy, dim=2)
                  
        attended_encoding = torch.matmul(attention, encoder_combined)
        attended_encoding = self.attn_emb2hid(attended_encoding)
        
        attended_combined = (conved + attended_encoding.permute(0, 2, 1)) * self.scale
        
        return attention, attended_combined
        
    def forward(self, trg, encoder_conved, encoder_combined):
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
            
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        tok_embedded = self.tok_embedding(trg)
        pos_embedded = self.pos_embedding(pos)
        
        embedded = self.dropout(tok_embedded + pos_embedded)        
        conv_input = self.emb2hid(embedded)
        conv_input = conv_input.permute(0, 2, 1) 
                
        batch_size = conv_input.shape[0]
        hid_dim = conv_input.shape[1]
        
        for i, conv in enumerate(self.convs):
        
            conv_input = self.dropout(conv_input)
            padding = torch.zeros(batch_size, 
                                  hid_dim, 
                                  self.kernel_size - 1).fill_(self.trg_pad_idx).to(self.device)
                
            padded_conv_input = torch.cat((padding, conv_input), dim = 2)
            
            conved = conv(padded_conv_input)
            conved = F.glu(conved, dim = 1)

            attention, conved = self.calculate_attention(embedded, 
                                                         conved, 
                                                         encoder_conved, 
                                                         encoder_combined)
                        
            conved = (conved + conv_input) * self.scale
            conv_input = conved
            
        conved = self.hid2emb(conved.permute(0, 2, 1))            
        output = torch.clamp(self.fc_out(self.dropout(conved)), MIN_VAL, MAX_VAL)
                    
        return output, attention

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, trg):
        
        encoder_conved, encoder_combined = self.encoder(src)
        output, attention = self.decoder(trg, encoder_conved, encoder_combined)
        
        return output, attention

SEQ_LENGTH = 128
BATCH_SIZE = 32
TRAIN_SIZE = 20000
VAL_SIZE = 4000
MIN_VAL = 0
MAX_VAL = 100
INPUT_DIM = OUTPUT_DIM = MAX_VAL - MIN_VAL + 1 

EMB_DIM = int(INPUT_DIM * 0.25)
HID_DIM = EMB_DIM * 2 
ENC_LAYERS = 12
DEC_LAYERS = 12
ENC_KERNEL_SIZE = 3
DEC_KERNEL_SIZE = 3 
ENC_DROPOUT = 0.25
DEC_DROPOUT = 0.25
TRG_PAD_IDX = MAX_VAL
CLIP = 0.5
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, ENC_LAYERS, ENC_KERNEL_SIZE, ENC_DROPOUT, device, SEQ_LENGTH)
dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, DEC_LAYERS, DEC_KERNEL_SIZE, DEC_DROPOUT, TRG_PAD_IDX, device, SEQ_LENGTH)

model = Seq2Seq(enc, dec).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def sq_euclidean_distance(y_true, y_pred):
    squared_difference = torch.square(y_true - y_pred)
    sq_distance = torch.mean(squared_difference, dim=-1) 
    return sq_distance

X_train = np.random.randint(low=MIN_VAL, high=MAX_VAL, size=(TRAIN_SIZE, SEQ_LENGTH))
Y_train = np.sort(X_train, axis=1)

X_val = np.random.randint(low=MIN_VAL, high=MAX_VAL, size=(VAL_SIZE, SEQ_LENGTH))
Y_val = np.sort(X_val, axis=1)

X_train_tensor = torch.Tensor(X_train)
Y_train_tensor = torch.Tensor(Y_train)
X_val_tensor = torch.Tensor(X_val)
Y_val_tensor = torch.Tensor(Y_val)

train_set = TensorDataset(X_train_tensor, Y_train_tensor)
val_set = TensorDataset(X_val_tensor, Y_val_tensor)
train_iterator = DataLoader(train_set, BATCH_SIZE)
valid_iterator = DataLoader(val_set, BATCH_SIZE)

def train(model, iterator, optimizer, clip):

    model.train()
    epoch_loss = 0
    
    for i, (X, Y) in enumerate(iterator):
        
        src = X.long().to(device)
        trg = Y.long().to(device)
        
        optimizer.zero_grad()
        output, _ = model(src, trg[:,:-1])
        output_dim = output.shape[-1]
        
        output = output.contiguous().view(-1, output_dim).squeeze()
        trg = trg[:,1:].contiguous().view(-1)

        loss = sq_euclidean_distance(output, trg.float())
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator):
    
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, (X, Y) in enumerate(iterator):
        
            src = X.long().to(device)
            trg = Y.long().to(device)

            output, _ = model(src, trg[:,:-1])
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim).squeeze()
            trg = trg[:,1:].contiguous().view(-1)

            loss = sq_euclidean_distance(output, trg.float())
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):

    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


###### TRAINING ######
# N_EPOCHS = 200
# CLIP = 0.1

# best_valid_loss = float('inf')

# for epoch in range(N_EPOCHS):
    
#     start_time = time.time()
    
#     train_loss = train(model, train_iterator, optimizer, CLIP)
#     valid_loss = evaluate(model, valid_iterator)
    
#     end_time = time.time()
    
#     epoch_mins, epoch_secs = epoch_time(start_time, end_time)

#     if valid_loss < best_valid_loss:
#         best_valid_loss = valid_loss
#         torch.save(model.state_dict(), f'seq2seq_model_v9_seq_32_{best_valid_loss:.3f}.pt')
    
#     print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
#     print(f'\tTrain Loss: {train_loss:.3f}')
#     print(f'\t Val. Loss: {valid_loss:.3f}')


state_dict = torch.load(f'seq2seq_model_v8_3.649.pt', map_location=torch.device('cpu'))
trained_model = Seq2Seq(enc, dec).to(device)
trained_model.load_state_dict(state_dict)

for i in range(10):

    sample_X = np.random.randint(MIN_VAL, MAX_VAL, size=(1, SEQ_LENGTH))
    sample_X_tensor = torch.LongTensor(sample_X).to(device)

    trg_indexes = [TRG_PAD_IDX]
    with torch.no_grad():
        encoder_conved, encoder_combined = model.encoder(sample_X_tensor)

        for i in range(SEQ_LENGTH):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
            with torch.no_grad():
                output, attention = model.decoder(trg_tensor, encoder_conved, encoder_combined)

            pred_token = output[0][-1].item()
            trg_indexes.append(pred_token)

    nn_output = trg_indexes[1:]
    adjusted_output = [np.asarray(sample_X).reshape(SEQ_LENGTH,)[np.abs(np.asarray(sample_X).reshape(SEQ_LENGTH,) - i).argmin()] for i in list(nn_output)]

    print("############################################################")
    print(sample_X[0].tolist(), ": Start Array")
    print(np.sort(sample_X)[0].tolist(), ": Target Array")
    print(adjusted_output, ": Adjusted Output")

    print("\n\n")