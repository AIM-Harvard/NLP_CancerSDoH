import torch
from test_tube import HyperOptArgumentParser
import numpy as np
from utils import grab_sections
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch.nn as nn
import torch.optim as optim
import string
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support
import tqdm
from transformers import AutoTokenizer, AutoModel
import random
from time import perf_counter
import pandas as pd
from collections import Counter
import json


parser = HyperOptArgumentParser(strategy='random_search')
parser.add_argument('--train_file', type=str, help='The path to train csv')
parser.add_argument('--dev_file', type=str, help='The path to dev csv')
parser.add_argument('--test_file', type=str, help='The path to test csv')
parser.add_argument('--logdir', type=str, help='The path to the directory to store model evaluation results')
parser.add_argument('--provider_type', type=str, help='provider type: "Physician" or "All_Providers')
parser.add_argument('--label', type=str, default='', help='Which category question to classify documents: race or gender')
parser.add_argument('--epochs', type=int, default=5, help='Number of training iterations')
parser.opt_list('--lr', type=float, help='Model learning rate', default=0.0001, tunable=True, options=[0.0001, 0.0005, .001, .003, .01])
parser.opt_list('--dropout', type=float, help='dropout for embedding layer', default=0.2, tunable=True, options=[0.1, 0.2, 0.5])
parser.opt_list('--seq_length', type=int, help='maximum sequence length which all docs will be padded / truncated', default=100, tunable=True, options=[100, 200, 300, 400, 500])
parser.opt_list('--batch_size', type=int, help='Batch size', default=32, tunable=True, options=[16, 32, 64, 128])
parser.add_argument('--model', type=str, help='select model to run classification: (BERT or BIOBERT)', default='BERT')
parser.add_argument('--undersample',  action='store_true', help='undersample majority class in train set')


args = parser.parse_args()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


class BertClassifier(nn.Module):

    def __init__(self, num_labels, model_type, dropout=0.1):

        super(BertClassifier, self).__init__()

        self.bert = AutoModel.from_pretrained(model_type)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_labels)

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        return linear_output


def preprocess(documents:list, labels:list, label_to_idx: dict) -> tuple[list, list]:
    """
    Get 'tokenized' documents and labels to corresponding integers
    """
    label_idx = [label_to_idx[lab] for lab in labels]
    tokenized_documents = []
    punc_set = set(string.punctuation)
    for doc in documents:
        doc = doc.split()
        tokens = [token for token in doc if token not in punc_set]
        tokenized_documents.append(tokens)
    return tokenized_documents, label_idx


def undersample(train_docs: list, train_labels: list) -> tuple[list, list]:
    """
    Reduce majority class training samples
    """
    label_counter = Counter(train_labels)
    print('Original Train Label Counts: ')
    print(label_counter)
    majority_class = label_counter.most_common()[0][0]
    minority_class = label_counter.most_common()[-1][0]
    majority_count = label_counter.most_common()[0][1]
    minority_count = label_counter.most_common()[-1][1]
    majority_docs = [doc for i, doc in enumerate(train_docs) if train_labels[i]==majority_class]
    minority_docs = [doc for i, doc in enumerate(train_docs) if train_labels[i]==minority_class]
    majority_labels = [doc for doc in train_labels if doc==majority_class]
    minority_labels = [doc for doc in train_labels if doc==minority_class]
    assert(len(majority_docs)==len(majority_labels)==majority_count)
    assert(len(minority_docs)==len(minority_labels)==minority_count)

    #shuffle majority class documents
    majority_temp = list(zip(majority_docs, majority_labels)) 
    random.shuffle(majority_temp) 
    maj_doc, maj_lab = zip(*majority_temp)
    maj_doc = list(maj_doc)
    maj_lab = list(maj_lab)
    maj_doc = maj_doc[:minority_count] #select majority class documents (n= #minority class docs)
    maj_lab = maj_lab[:minority_count]
    train_labels ,train_docs = [], []
    train_labels.extend(minority_labels)
    train_labels.extend(maj_lab)
    train_docs.extend(minority_docs)
    train_docs.extend(maj_doc)
    new_counter = Counter(train_labels)
    print('New Train Label Counts: ')
    print(new_counter)
    return train_docs, train_labels

def tokenize_atten(sentences: list, labels: list, tokenizer: AutoTokenizer, max_length: int):    
    """
    Tokenize all of the sentences and map the tokens to thier word IDs
    Return input_ids, attention masks, and labels all as tensors
    """
    input_ids = []
    attention_masks = []
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,                     
                            add_special_tokens = True, 
                            max_length = max_length,           
                            padding='max_length',
                            return_attention_mask = True,
                            return_tensors = 'pt', 
                            truncation = True
                    )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    return input_ids, attention_masks, labels

def finetune_bert(train_dataloader: DataLoader, epochs: int, val_loader: DataLoader, loss: nn.CrossEntropyLoss, optimizer:optim.Adam, model, label_dict):
    """
    BERT classifier training loop
    """
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm.tqdm(train_dataloader, desc=f'[Training {epoch+1}/{epochs}]'):
            optimizer.zero_grad()
            b_input_ids = batch[0].to(DEVICE)
            b_input_mask = batch[1].to(DEVICE)
            b_labels = batch[2].to(DEVICE)
            output = model(b_input_ids, b_input_mask)
            
            batch_loss = loss(output, b_labels)
            epoch_loss += batch_loss.item()
        
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, epoch_loss))
        _ = evaluate_bert(val_loader, model, label_dict)
    _ = evaluate_bert(val_loader, model, label_dict) # <---- DELETE


def evaluate_bert(val_loader: DataLoader, model, label_dict) -> dict:
    """
    Evaluate on test/dev sets
    Return dict of metrics
    """
    preds = []
    probas = []
    golds = []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for batch in tqdm.tqdm(val_loader, desc=f'[Validation]'):
            b_input_ids = batch[0].to(DEVICE)
            b_input_mask = batch[1].to(DEVICE)
            b_labels = batch[2].to(DEVICE)
            logits = model(b_input_ids, b_input_mask)
            logits = logits.detach().cpu().numpy()
            labels = b_labels.to('cpu').numpy()
            log_temp = torch.tensor(logits)
            sm_out = softmax(log_temp)
            probas.append(sm_out)
            preds.append(logits)
            golds.append(labels)
    all_probas = np.concatenate(probas, axis=0)
    pos_class = all_probas[:,1]
    all_logits = np.concatenate(preds, axis=0)
    all_golds = np.concatenate(golds, axis=0)
    metrics = classification_eval_metrics(all_logits, all_golds, label_dict)
    metrics['AUC'] = roc_auc_score(all_golds, pos_class)
    return metrics 


def classification_eval_metrics(preds, labels, idx2lab):
    """
    Calculate Evaluatuin metrics
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    all_preds = pred_flat.tolist()
    all_golds = labels_flat.tolist()
    all_labels = [idx2lab[true_idx] for true_idx in all_golds]
    all_predicitions = [idx2lab[pred_idx] for pred_idx in all_preds]
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_predicitions)
    micro_f1  = precision_recall_fscore_support(all_labels, all_predicitions, average='micro')[2]
    weight_f1 = precision_recall_fscore_support(all_labels, all_predicitions, average='weighted')[2]
    macro_f1 = precision_recall_fscore_support(all_labels, all_predicitions, average='macro')[2]
    p_class0 = prec[0]
    p_class1 = prec[1]
    r_class0 = rec[0]
    r_class1 = rec[1]
    f_class0 = f1[0]
    f_class1 = f1[1]
    print(classification_report(all_labels, all_predicitions))
    return {
        'macro_f1':macro_f1, 
        'micro_f1': micro_f1,
        'weighted_f1': weight_f1, 
        'precision_0': p_class0,
        'precision_1': p_class1,
        'recall_0': r_class0,
        'recall_1':r_class1,
        'f1_0': f_class0,
        'f1_1': f_class1
        }


if __name__ =='__main__':
    load_start = perf_counter()
    train_df = pd.read_csv(args.train_file, encoding='utf8')
    dev_df = pd.read_csv(args.dev_file, encoding='utf8')
    test_df = pd.read_csv(args.test_file, encoding='utf8')

    print('='*30)
    print('Data Loaded in {:.2f} seconds'.format(perf_counter()-load_start))
    print('='*30)
    if args.label == 'Race_group':
        train_df.loc[(train_df['Race_group']!='White_NonHispanic', 'Race_group')] = 'NonWhite'
        dev_df.loc[(dev_df['Race_group']!='White_NonHispanic', 'Race_group')] = 'NonWhite'
        test_df.loc[(test_df['Race_group']!='White_NonHispanic', 'Race_group')] = 'NonWhite'
    label_to_idx = {label:i for i, label in enumerate(list(set(train_df[args.label])))}
    idx_to_label = {v:k for k, v in label_to_idx.items()}

    train_labels = train_df[args.label]
    val_labels = dev_df[args.label]
    test_labels = test_df[args.label]

    train_docs = train_df['text']
    val_docs = dev_df['text']
    test_docs = test_df['text']
    prep_start = perf_counter()

    if args.undersample:
        train_docs, train_labels = undersample(train_docs, train_labels)
    train_docs = [grab_sections(note, token_len=300).lower() for note in train_docs]
    val_docs = [grab_sections(note, token_len=300).lower() for note in val_docs]
    test_docs = [grab_sections(note, token_len=300).lower() for note in test_docs]

    train_tokenized, train_y = preprocess(train_docs, train_labels, label_to_idx)
    dev_tokenized, dev_y = preprocess(val_docs, val_labels, label_to_idx)
    test_tokenized, test_y = preprocess(test_docs, test_labels, label_to_idx)

    if args.model == 'BERT':
        pretrained_model = 'bert-base-uncased'
    elif args.model == 'BIOBERT':
        pretrained_model = 'emilyalsentzer/Bio_ClinicalBERT'
    else:
        raise Exception('MUST HAVE VALID MODEL: BERT OR BIOBERT')
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    dev_sents = [' '.join(dev_sent) for dev_sent in dev_tokenized]
    train_sents = [' '.join(train_sent) for train_sent in train_tokenized]
    test_sents = [' '.join(test_sent) for test_sent in test_tokenized]

    val_input_ids, val_attention_masks, val_labels = tokenize_atten(dev_sents, dev_y, tokenizer, args.seq_length)
    train_input_ids, train_attention_masks, train_labels = tokenize_atten(train_sents, train_y, tokenizer, args.seq_length)
    test_input_ids, test_attention_masks, test_labels = tokenize_atten(test_sents, test_y, tokenizer, args.seq_length)

    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
    val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)
    train_loader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = args.batch_size)
    dev_loader = DataLoader(val_dataset, sampler = SequentialSampler(val_dataset), batch_size = args.batch_size)
    test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)
    test_loader = DataLoader(test_dataset, sampler = RandomSampler(test_dataset), batch_size = args.batch_size)
    print('='*30)        
    print('Preprocessed in {:.2f} seconds'.format(perf_counter()-prep_start))
    print('='*30)
    model = BertClassifier(num_labels=len(label_to_idx), model_type=pretrained_model, dropout=args.dropout)
    model.to(DEVICE)
    print('Device: ',DEVICE)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_start = perf_counter()
    print('==============TRAINING===============')
    finetune_bert(train_loader, args.epochs, dev_loader, loss, optimizer, model, idx_to_label)
    print('='*30)
    print('Training done in {:.2f} seconds'.format(perf_counter()-train_start))
    print('='*30)

    test_set_eval = evaluate_bert(test_loader, model, idx_to_label)
    # torch.save(model, './model_dir/'+args.model+'_'+args.provider_type+'_'+args.label+'.pth')
    with open(args.logdir+'/'+args.model+'_'+args.provider_type+'_'+args.label+'_results.json', 'w', encoding='utf8') as out_f:
        json.dump(test_set_eval, out_f)
