    ### TODO: Create any instance variables you need to classify the sentiment of BERT embeddings.
    ### YOUR CODE HERE
    # dropout before linear
    self.dropout = torch.nn.Dropout(0.3)
    # need final linear layer to map hidden size to number of classes
    self.last_linear_layer = torch.nn.Linear(self.gpt.config.hidden_size, self.num_labels)


  def forward(self, input_ids, attention_mask):
    '''Takes a batch of sentences and returns logits for sentiment classes'''

    ### TODO: The final GPT contextualized embedding is the hidden state of [CLS] token (the first token).
    ###       HINT: You should consider what is an appropriate return value given that
    ###       the training loop currently uses F.cross_entropy as the loss function.
    ### YOUR CODE HERE
    # should be last token, which has seen all previous tokens
    #logit = self.gpt.hidden_state_to_token(self.gpt(input_ids, attention_mask)['last_token'])
    # map from logit: hidden_size to number of classes for sentiment classification
    last_token_hidden = self.gpt(input_ids, attention_mask)['last_token']
    
    return self.last_linear_layer(self.dropout(last_token_hidden))