import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
        
class PretrainModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(PretrainModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = nn.Linear(config.hidden_size, 640)
        self.args = args

        
    def forward(self, inputs_ids, position_idx, csg_edge_mask, labels=None):

        inputs_embeddings=self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        
        outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings, position_ids=position_idx, token_type_ids=position_idx.eq(-1).long())[0]
        logits=self.classifier(outputs)                
        probability=F.softmax(logits, dim=-1)

        loss_function = CrossEntropyLoss()
        device = csg_edge_mask.device
        csg_edge_mask = torch.where(csg_edge_mask, torch.tensor(1).to(device), torch.tensor(0).to(device)).float()
        loss = loss_function(logits, csg_edge_mask)
        return loss, probability
