import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

class RobertaClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take [CLS] token, torch.Size([batch, 768])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class PVModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(PVModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args

        
    def forward(self, inputs_ids, position_idx, csg_edge_mask, labels=None):
        # csg_node_mask = csg_d_mask + csg_c_mask + csg_v_mask
        csg_node_mask = position_idx.eq(0)                    # csg_node_mask:  torch.Size([batch, 640=512+128=code_length+csg_node_length])
        token_mask = position_idx.ge(2)                       # token_mask:  torch.Size([batch, 640])
        inputs_embeddings = self.encoder.roberta.embeddings.word_embeddings(inputs_ids)   # inputs_embeddings:  torch.Size([batch, 640, 768])
        csg_to_token_mask = csg_node_mask[:,:,None]&token_mask[:,None,:]&csg_edge_mask      # csg_to_token_mask:  torch.Size([batch, 640, 640])
        csg_to_token_mask = csg_to_token_mask / (csg_to_token_mask.sum(-1) + 1e-10)[:, :, None] 
        csg_avg_embeddings = torch.einsum("abc,acd->abd", csg_to_token_mask, inputs_embeddings) 
        inputs_embeddings = inputs_embeddings*(~csg_node_mask)[:,:,None]+csg_avg_embeddings*csg_node_mask[:,:,None]
        
        outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings,attention_mask=csg_edge_mask,position_ids = position_idx,token_type_ids=position_idx.eq(-1).long())[0]
        logits = self.classifier(outputs)                 # logits:  torch.Size([batch, num_classes=2])
        probability = F.softmax(logits, dim=-1)
        if labels is not None:
            loss_function = CrossEntropyLoss()
            loss = loss_function(logits, labels)
            return loss,probability
        else:
            return probability
