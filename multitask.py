import torch.nn as nn
from pytorch_transformers import (
    BertForSequenceClassification, BertTokenizer
)

class BertForMultitaskClassification(BertForSequenceClassification):
    def __init__(self, config):
        assert hasattr(config, "num_labels_per_task")
        assert sum(config.num_labels_per_task) == config.num_labels
        super().__init__(config)
        self.num_tasks = len(config.num_labels_per_task)

    def loss_fct(self, logits, labels):
        loss = 0
        inner_loss_fct = nn.CrossEntropyLoss(reduction="none")
        offset = 0
        task_masks = labels[:, 1:].float() # this conversion is inefficient...
                                           # TODO: if this turns out to be slow, optimize
        for task_id, nl in enumerate(self.config.num_labels_per_task):
            task_loss = inner_loss_fct(logits[:, offset:offset+nl],
                                       labels[:, 0])
            loss += (task_loss * task_masks[:, task_id]).mean()
            offset += nl
        return loss

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1, 1+self.num_tasks))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
