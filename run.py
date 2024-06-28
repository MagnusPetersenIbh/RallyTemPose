import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange, repeat
import os 
import argparse
from sklearn.metrics import accuracy_score
from Models.nn_models import RallyTempose
from Utils.tools import *
from Utils.playerID_utils import *
from collections import defaultdict
from sklearn.model_selection import train_val_split
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import pandas as pd

def main(args):
    if args.clip_grad == 1:
        clip_grad = True
    else:
        clip_grad = False
    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.

    
    load_model = False
    save_model = True
    include_pos = True
    model_filename= args.file_save#"best_model_shutset_rallytemposeII.pt" ### arg.model_checkpoint_file
    splitting = args.split
    #Load data
  
    poses = pd.read_pickle('./Data/shuttleset/2d_poses_shuttleset_cont.pkl')
    labels = pd.read_pickle('./Data/shuttleset/2d_labels_shuttleset_cont.pkl')
    position = pd.read_pickle('./Data/shuttleset/2d_positions_shuttleset_cont.pkl')
    match_info = pd.read_csv('./Data/shuttleset/match.csv')
    
    batch_size = 1
    import scipy
    if batch_size!=1:
        do_pad = True
    else:
        do_pad = False

    
    include_pos=True
    grouped_data, encoded_shot_labels,grouped_position,grouped_playerID = group_and_encode_shuttlenet_data_with_player(poses, labels,position,match_info,padding_shift=do_pad,convert=True,smoothing=False,player_splitting=True)

    org_by_match = stack_grouped_data_into_lists(grouped_data, encoded_shot_labels,grouped_position,grouped_playerID)
    seq_len = 30
    min_len = 3

    
    data_train_raw, data_val_raw, target_train_raw, target_val_raw,pos_train_raw,pos_val_raw,Id_train_raw,Id_val_raw = prop_match_test_train_split(org_by_match,ratio= 0.20,r_state=splitting)
    data_train_raw, target_train_raw, pos_train_raw,Id_train_raw = filter_sequences_by_lengthID(data_train_raw, target_train_raw, pos_train_raw,Id_train_raw, min_len, seq_len)
    data_val_raw, target_val_raw, pos_val_raw,Id_val_raw = filter_sequences_by_lengthID(data_val_raw, target_val_raw, pos_val_raw,Id_val_raw, min_len, seq_len)
    
    #Model inputs def
    channels = 4
    num_joints = 17# num keypoints (16) + position optionally (1)
    trg_vocab_size = 10
    
    num_epochs = 300 
    warmup_e_pr = 0.3
    learning_rate = args.LR
    learning_rate_min = 1e-6


    embedding_scale =16
    num_heads = 4 
    num_encoder_layers = args.T_layers 
    num_spat_layers =  args.T_layers 
    num_decoder_layers =  args.N_layers 
    dropout_attn = args.dropout_a 
    dropout_proj = args.dropout_e 
    dropout_embed = args.dropout_e 
    drop_path = args.drop_path 
    max_len = 30 
    embedding_scale_spat = 6
    embedding_size = embedding_scale*num_heads
    embed_size_spatial = embedding_scale_spat*num_heads
    
    forward_expansion = 4
    
    trg_pad_idx = -1 ## artifact from testing seqeunce padding
    warmup_e = int(warmup_e_pr*num_epochs)
    

    # Train and val splits
    # data loading
    shift = 0
    do_aux = True
    

    train_data = PoseData_Forecast(data_train_raw, target_train_raw, pos=pos_train_raw,playerID=Id_train_raw,len_max=max_len,factorized=True,multi_con=True,tjek_enc=False)
    val_data = PoseData_Forecast(data_val_raw, target_val_raw, pos = pos_val_raw,playerID=Id_val_raw,len_max=max_len,factorized=True,multi_con=True,tjek_enc=False)

    

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    model = RallyTempose(
        embedding_size,
        embed_size_spatial,
        channels,
        num_joints,
        trg_vocab_size,
        trg_pad_idx,
        num_heads,
        num_heads,
        num_encoder_layers,
        num_spat_layers,
        num_decoder_layers,
        forward_expansion,
        dropout_attn,
        dropout_proj,
        dropout_embed,
        drop_path,
        max_len,
        35,
        prob_bool=do_aux
    )
    parallel = False 
    ids = [1,3]
    if parallel:
        model = nn.DataParallel(model, device_ids=ids)
        model = model.to(device)
    else:
        model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

    aux_criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
  
   
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay = 0.001)
    
    criterion = criterion.to(device)

    best_accuracy = load_best_model(model, optimizer,filename=model_filename)


    print(f'starting training..')
    
    accumulation_steps = args.pseudo_bs  # This means the effective batch size is batch_size * accumulation_steps ## if wanted .. 
    aux_ratio = args.aux_r
    best_loss = float('inf')
    patience = 200
    epochs_no_improve = 0
    shift = 0 
    ignore_first = 1  # i.e has 2 prior actions for evaluated predictions 
    acc_this_run= 0
    acc2_this_run = 0
    acc3_this_run = 0 
    best_epoch_this_run = 0
    bert_model_name = 'bert-base-uncased'
    #tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    print_int = 25
    ig_error_id = 9
    for epoch in range(num_epochs):
    
        model.train()
        #
        epLoss = 0
        epAcc = 0
        epauxAcc = 0
        adjust_lr(optimizer,learning_rate,learning_rate_min,epoch,num_epochs,warmup_e)
        optimizer.zero_grad()
        for i,(x, y,pad,x_id,text) in enumerate(train_loader): ### Fetch a batch of training inputs
            x = x.type(torch.FloatTensor).to(device)
            y = y.to(device)
            x_id = x_id.to(device)
            pad = pad.to(device)
            if bert_model_name is not None:
                encoded_input = text['input_ids'].to(device)
                bert_mask = text['attention_mask'].to(device)
                
            else:
                encoded_input = y.copy()
                bert_mask = None          
            
            if do_aux:
                yPred,aux_pred = model(x[:,shift:-1], encoded_input[:, shift:-1],x_id[:,shift:-1],pad[:,shift:-1],bert_mask=bert_mask[:,shift:-1])
                loss1 = criterion(yPred[:,ignore_first:].reshape(-1, yPred.shape[2]), y[:,1+ignore_first+shift:].reshape(-1))
                loss2 = aux_criterion(aux_pred[:,:].reshape(-1, aux_pred.shape[2]), y[:,shift:-1].reshape(-1))
                loss = loss1 + aux_ratio*loss2
                
            else: 
                yPred = model(x[:,shift:-1], encoded_input[:, shift:-1],x_id[:,shift:-1],pad[:,shift:-1],bert_mask=bert_mask[:,shift:-1])
                loss = criterion(yPred[:,ignore_first:,].reshape(-1, yPred.shape[2]), y[:,1+ignore_first+shift:].reshape(-1))
            loss = loss / accumulation_steps  # Normalize our loss (if averaged)
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            ### Backpropagation steps
            ### Clear old gradients
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()  # Perform a step by updating the weights
                optimizer.zero_grad()  # Clear gradients for the next accumulation

        
            epLoss += loss.item()
            if parallel:
                preds = model.module.predict(x[:,shift:-1], y[:, shift:-1],x_id[:,shift:-1],pad[:,shift:-1],bert_mask=bert_mask[:,shift:-1])[:,ignore_first:].reshape(-1)
            else:
                if do_aux:
                    preds,aux_guees = model.predict(x[:,shift:-1], encoded_input[:, shift:-1],x_id[:,shift:-1],pad[:,shift:-1],bert_mask=bert_mask[:,shift:-1])
                    preds = preds[:,ignore_first:].reshape(-1)
                    aux_guees = aux_guees.reshape(-1)
                    target1 = y[:,1+ignore_first+shift:].reshape(-1)
                    target2 = y[:,shift:-1].reshape(-1)
                    
                    acc = custom_accuracy_score(preds.cpu().numpy(),target1.cpu().numpy())#,trg_pad_idx) ## also implement top3 acc score
                    aux_acc = custom_accuracy_score(aux_guees.cpu().numpy(),target2.cpu().numpy())#,trg_pad_idx)
                    epauxAcc += aux_acc
                else:
                    preds = model.predict(x[:,shift:-1], y[:, shift:-1],x_id[:,shift:-1],pad[:,shift:-1],bert_mask=bert_mask[:,shift:-1])[:,ignore_first:].reshape(-1)

                    target = y[:,1+ignore_first+shift:].reshape(-1)
                    
                    acc = custom_accuracy_score(preds.cpu().numpy(),target.cpu().numpy())#,trg_pad_idx) ## also implement top3 acc score
                    
            epAcc += acc
        if epoch%print_int==0:
            print(f'train acc: {epAcc/len(train_loader)}')
            print(f'trian loss: {epLoss/len(train_loader)}')
            if do_aux:
                print(f'val aux task acc: {epauxAcc/len(train_loader)}')

        ### VALIDATION

        epLoss = 0
        if do_aux:
            epauxAcc = 0
        epAcc = 0
        epAcc2 = 0
        epAcc3 = 0
        #
        optimizer.zero_grad()
        model.eval()
        
        for x, y,pad,x_id,text in val_loader: #### Fetch validation samples
            x = x.type(torch.FloatTensor).to(device)
            y = y.to(device)
            x_id = x_id.to(device)
            
            pad = pad.to(device)

            if bert_model_name is not None:
                encoded_input = text['input_ids'].to(device)
                
                bert_mask = text['attention_mask'].to(device)
                
            else:
                encoded_input = y.copy()
                bert_mask = None
            if do_aux:
                yPred,aux_pred = model(x[:,shift:-1], encoded_input[:, shift:-1],x_id[:,shift:-1],pad[:,shift:-1],bert_mask=bert_mask[:,shift:-1])
                loss1 = criterion(yPred[:,ignore_first:].reshape(-1, yPred.shape[2]), y[:,1+ignore_first+shift:].reshape(-1))
                loss2 = aux_criterion(aux_pred[:,:].reshape(-1, aux_pred.shape[2]), y[:,shift:-1].reshape(-1))
                loss = loss1 + aux_ratio*loss2
            else: 
                yPred = model(x[:,shift:-1], y[:, shift:-1],x_id[:,shift:-1],pad[:,shift:-1],bert_mask=bert_mask[:,shift:-1])
                loss = criterion(yPred[:,ignore_first:,].reshape(-1, yPred.shape[2]), y[:,1+ignore_first+shift:].reshape(-1))

            epLoss += loss.item()
            if parallel:
                preds = model.module.predict(x[:,shift:-1], y[:, shift:-1],x_id[:,shift:-1],pad[:,shift:-1],bert_mask=bert_mask[:,shift:-1])[:,ignore_first:].reshape(-1) ## not used atm
            else:
                if do_aux:
                    preds,aux_guees = model.predict(x[:,shift:-1], encoded_input[:, shift:-1],x_id[:,shift:-1],pad[:,shift:-1],bert_mask=bert_mask[:,shift:-1])
                    preds = preds[:,ignore_first:].reshape(-1)
                    aux_guees = aux_guees.reshape(-1)
                    target1 = y[:,1+ignore_first+shift:].reshape(-1)
                    target2 = y[:,shift:-1].reshape(-1)
                    #acc = accuracy_score(preds.cpu().numpy(),target.cpu().numpy())#accuracy(y,yPred)
                    acc = custom_accuracy_score(preds.cpu().numpy(),target1.cpu().numpy())#,trg_pad_idx) ## also implement top3 acc score
                    acc2 = custom_top_k_accuracy(yPred[:,ignore_first:].reshape(-1, yPred.shape[2]).cpu().detach(),y[:,1+ignore_first+shift:].reshape(-1).cpu().detach(),k=2)#,ignore_index=9)
                    acc3 = custom_top_k_accuracy(yPred[:,ignore_first:].reshape(-1, yPred.shape[2]).cpu().detach(),y[:,1+ignore_first+shift:].reshape(-1).cpu().detach(),k=3)#,ignore_index=9)
                    aux_acc = custom_accuracy_score(aux_guees.cpu().numpy(),target2.cpu().numpy())#,trg_pad_idx)
                    epauxAcc += aux_acc
                else:
                    preds = model.predict(x[:,shift:-1], y[:, shift:-1],x_id[:,shift:-1],pad[:,shift:-1],bert_mask=bert_mask[:,shift:-1])[:,ignore_first:].reshape(-1)

                    target = y[:,1+ignore_first+shift:].reshape(-1)
                    #acc = accuracy_score(preds.cpu().numpy(),target.cpu().numpy())#accuracy(y,yPred)
                    acc = custom_accuracy_score(preds.cpu().numpy(),target.cpu().numpy(),ig_error_id) ## also implement top3 acc score
                    acc2 = custom_top_k_accuracy(yPred[:,ignore_first:].reshape(-1, yPred.shape[2]).cpu().detach(),y[:,1+ignore_first+shift:].reshape(-1).cpu().detach(),k=2)#,ignore_index=9)
                    acc3 = custom_top_k_accuracy(yPred[:,ignore_first:].reshape(-1, yPred.shape[2]).cpu().detach(),y[:,1+ignore_first+shift:].reshape(-1).cpu().detach(),k=3)#,ignore_index=9)

            epAcc += acc
            epAcc2 += acc2
            epAcc3 += acc3
        if epoch%print_int==0:
            print(f'epoch {epoch}: val acc: {epAcc/len(val_loader)} ## val acc top 2: {epAcc2/len(val_loader)} ## val acc top 3: {epAcc3/len(val_loader)} ## val loss: {epLoss/len(val_loader)}')
            if do_aux:
                print(f'val aux task acc: {epauxAcc/len(val_loader)}')

        currentLoss = epLoss/len(val_loader)
        accuracy = epAcc/len(val_loader)
        # 

        if accuracy>acc_this_run:
            acc_this_run = accuracy
            acc2_this_run = epAcc2/len(val_loader)
            acc3_this_run = epAcc3/len(val_loader)
            best_epoch_this_run = epoch
            print(f'new best acc this run:{acc_this_run} at epoch {epoch}')

        if currentLoss < best_loss:
            best_loss = currentLoss
            epochs_no_improve = 0

        else:
            epochs_no_improve += 1

        if epoch > warmup_e and epochs_no_improve >= patience:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Model Configuration")

    # Add command-line arguments for model configuration
    parser.add_argument("--T_layers", type=int, default=2, help="Number of temporal transformer layers")
    parser.add_argument("--N_layers", type=int, default=2, help="Number of Interaction transformer layers")
    parser.add_argument("--file_save", type=str, default='best_model.pt', help="str name to same model param")
    parser.add_argument("--split", type=int, default=12, help="Data splitting")
    parser.add_argument("--dropout_a", type=float, default=0.3, help="dropout attention ratio during training")
    parser.add_argument("--dropout_e", type=float, default=0.3, help="dropout embed ratio during training")
    parser.add_argument("--drop_path", type=float, default=0.0, help="drop path rate during training")
    parser.add_argument("--aux_r", type=float, default=0.3, help="aux training parameter")
    parser.add_argument("--LR", type=float, default=0.00005, help="LR during training")
    parser.add_argument("--clip_grad", type=int, default=1, help="clipping the gradient")
    parser.add_argument("--pseudo_bs", type=int, default=4, help="sum gradients")

    args = parser.parse_args()
    main(args)
