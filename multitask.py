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
        inner_loss_fct = nn.CrossEntropyLoss()
        offset = 0
        for task_id, nl in enumerate(self.config.num_labels_per_task):
            loss += inner_loss_fct(logits[:, offset:offset+nl], labels[:, task_id]) / self.num_tasks
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
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_tasks))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
