#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N Winter 2025: Homework 3
nmt_model.py: NMT Model
"""

from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model_embeddings import ModelEmbeddings

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class NMT(nn.Module):
    """ Simple Neural Machine Translation Model:
        - Bidrectional LSTM Encoder
        - Unidirection LSTM Decoder
        - Global Attention Model (Luong, et al. 2015)
    """

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2):
        """ Init NMT Model.

        @param embed_size (int): Embedding size (dimensionality)
        @param hidden_size (int): Hidden Size, the size of hidden states (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        @param dropout_rate (float): Dropout probability, for attention
        """
        super(NMT, self).__init__()
        self.model_embeddings = ModelEmbeddings(embed_size, vocab)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab

        # default values
        self.post_embed_cnn = None
        self.encoder = None
        self.decoder = None
        self.h_projection = None
        self.c_projection = None
        self.att_projection = None
        self.combined_output_projection = None
        self.target_vocab_projection = None
        self.dropout = None
        # For sanity check only, not relevant to implementation
        self.gen_sanity_check = False
        self.counter = 0


        ### YOUR CODE HERE (~9 Lines)
        ### TODO - Initialize the following variables IN THIS ORDER:
        ###     self.post_embed_cnn (Conv1d layer with kernel size 2, input and output channels = embed_size,
        ###         padding = same to preserve output shape )
        ###     self.encoder (Bidirectional LSTM with bias)
        ###     self.decoder (LSTM Cell with bias)
        ###     self.h_projection (Linear Layer with no bias), called W_{h} in the PDF.
        ###     self.c_projection (Linear Layer with no bias), called W_{c} in the PDF.
        ###     self.att_projection (Linear Layer with no bias), called W_{attProj} in the PDF.
        ###     self.combined_output_projection (Linear Layer with no bias), called W_{u} in the PDF.
        ###     self.target_vocab_projection (Linear Layer with no bias), called W_{vocab} in the PDF.
        ###     self.dropout (Dropout Layer)
        ###
        ### Use the following docs to properly initialize these variables:
        ###     Conv1d:
        ###         https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        ###     LSTM:
        ###         https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        ###     LSTM Cell:
        ###         https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html
        ###     Linear Layer:
        ###         https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        ###     Dropout Layer:
        ###         https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html

        # padding same keeps input and output dim same, 1d conv act like reweighting here
        self.post_embed_cnn = nn.Conv1d(in_channels=embed_size, out_channels=embed_size, kernel_size=2, padding = 'same')
        # the bidirectional LSTM
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=self.hidden_size, num_layers=1, bias=True, dropout=self.dropout_rate, bidirectional=True)
        # the LSTM cell for decoder, combine embedding of the t-th word with output of t-1 step, resulting in input size of embedsize+hiddensize
        self.decoder = nn.LSTMCell(input_size=self.hidden_size+embed_size, hidden_size=self.hidden_size, bias=True)
        # h and c projection for the initial h and c in the decoder, mapping 2h dim to 1h dim
        self.h_projection = nn.Linear(in_features=2*self.hidden_size, out_features=self.hidden_size, bias=False)
        self.c_projection = nn.Linear(in_features=2*self.hidden_size, out_features=self.hidden_size, bias=False)
        # attention projection, just like W(key) in attention, h_dec is query, h_enc is key
        self.att_projection = nn.Linear(in_features=2*self.hidden_size, out_features=self.hidden_size, bias=False)
        # combined output projection: combined affinity reweighted bidirection encoder hidden state with decoder hidden state, then linear project
        self.combined_output_projection = nn.Linear(in_features=3*self.hidden_size, out_features=self.hidden_size, bias=False)
        # from h to V 
        self.target_vocab_projection = nn.Linear(in_features=self.hidden_size, out_features=len(self.vocab.tgt), bias=False)
        self.dropout = nn.Dropout(p=self.dropout_rate)

        ### END YOUR CODE

    def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
        """ Take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences under the language models learned by the NMT system.

        @param source (List[List[str]]): list of source sentence tokens
        @param target (List[List[str]]): list of target sentence tokens, wrapped by `<s>` and `</s>`

        @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        """
        # Compute sentence lengths
        source_lengths = [len(s) for s in source]

        # Convert list of lists into tensors
        source_padded = self.vocab.src.to_input_tensor(source, device=self.device)  # Tensor: (src_len, b)
        target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)  # Tensor: (tgt_len, b)

        ###     Run the network forward:
        ###     1. Apply the encoder to `source_padded` by calling `self.encode()`
        ###     2. Generate sentence masks for `source_padded` by calling `self.generate_sent_masks()`
        ###     3. Apply the decoder to compute combined-output by calling `self.decode()`
        ###     4. Compute log probability distribution over the target vocabulary using the
        ###        combined_outputs returned by the `self.decode()` function.

        enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths)
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)
        #print("target padded before decode shape", target_padded.shape)
        #print("combined_outputs shape", combined_outputs.shape)
        P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)
        #print("probability shape", P.shape)

        # Zero out, probabilities for which we have nothing in the target text
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()

        # Compute log probability of generating true target words
        assert P.shape[0] >= target_padded[1:].shape[0]
        target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(
            -1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum(dim=0)
        return scores

    def encode(self, source_padded: torch.Tensor, source_lengths: List[int], grader_params=None) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """ Apply the encoder to source sentences to obtain encoder hidden states.
            Additionally, take the final states of the encoder and project them to obtain initial states for decoder.

        @param source_padded (Tensor): Tensor of padded source sentences with shape (src_len, b), where
                                        b = batch_size, src_len = maximum source sentence length. Note that
                                       these have already been sorted in order of longest to shortest sentence.
        @param source_lengths (List[int]): List of actual lengths for each of the source sentences in the batch
        @returns enc_hiddens (Tensor): Tensor of hidden units with shape (b, src_len, h*2), where
                                        b = batch size, src_len = maximum source sentence length, h = hidden size.
        @returns dec_init_state (tuple(Tensor, Tensor)): Tuple of tensors representing the decoder's initial
                                                hidden state and cell.
        @grader_params: Ignore this parameter. It is used for grading purposes.
        """
        enc_hiddens, dec_init_state = None, None

        ### YOUR CODE HERE (~ 11 Lines)
        ### TODO:
        ###     1. Construct Tensor `X` of source sentences with shape (src_len, b, e) using the source model embeddings.
        ###         src_len = maximum source sentence length, b = batch size, e = embedding size. Note
        ###         that there is no initial hidden state or cell for the encoder.
        ###     2. Apply the post_embed_cnn layer. Before feeding X into the CNN, first use torch.permute to change the
        ###         shape of X to (b, e, src_len). After getting the output from the CNN, remember to use torch.permute
        ###         again to revert X back to its original shape.
        ###     3. Compute `enc_hiddens`, `last_hidden`, `last_cell` by applying the encoder to `X`.
        ###         - Before you can apply the encoder, you need to apply the `pack_padded_sequence` function to X.
        ###         - After you apply the encoder, you need to apply the `pad_packed_sequence` function to enc_hiddens.
        ###         - Note that the shape of the tensor returned by the encoder is (src_len, b, h*2) and we want to
        ###           return a tensor of shape (b, src_len, h*2) as `enc_hiddens`.
        ###     4. Compute `dec_init_state` = (init_decoder_hidden, init_decoder_cell):
        ###         - `init_decoder_hidden`:
        ###             `last_hidden` is a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards.
        ###             Concatenate the forwards and backwards tensors to obtain a tensor shape (b, 2*h).
        ###             Apply the h_projection layer to this in order to compute init_decoder_hidden.
        ###             This is h_0^{dec} in the PDF. Here b = batch size, h = hidden size
        ###         - `init_decoder_cell`:
        ###             `last_cell` is a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards.
        ###             Concatenate the forwards and backwards tensors to obtain a tensor shape (b, 2*h).
        ###             Apply the c_projection layer to this in order to compute init_decoder_cell.
        ###             This is c_0^{dec} in the PDF. Here b = batch size, h = hidden size
        ###
        ### See the following docs, as you may need to use some of the following functions in your implementation:
        ###     Pack the padded sequence X before passing to the encoder:
        ###         https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
        ###     Pad the packed sequence, enc_hiddens, returned by the encoder:
        ###         https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_packed_sequence.html
        ###     Tensor Concatenation:
        ###         https://pytorch.org/docs/stable/generated/torch.cat.html
        ###     Tensor Permute:
        ###         https://pytorch.org/docs/stable/generated/torch.permute.html

        # form Tensor X
        # source_padded: [src_len, b]

        X = self.model_embeddings.source(source_padded)
        assert len(list(X.shape)) == 3 and X.shape[0] == source_padded.shape[0] and X.shape[1] == source_padded.shape[1] and X.shape[2] == self.model_embeddings.embed_size
       
        # permute X s.t shape is batch, embed_size, src_length since 1d conv act on the leading dimension]
        X_permuted = torch.permute(X, (1, 2, 0))

        # pass through 1d conv then permute back into src_length, batch, embed size
        X_permuted_post_cnn = self.post_embed_cnn(X_permuted)

        X_post_cnn = torch.permute(X_permuted_post_cnn, (2, 0, 1))
        assert X_post_cnn.shape[0] == X.shape[0] and X_post_cnn.shape[1] == X.shape[1] and X_post_cnn.shape[2] == X.shape[2] 
        

        # pack tensor containing padded sequences of variable length
        X_packed = pack_padded_sequence(input=X_post_cnn, lengths=source_lengths, batch_first=False, enforce_sorted=True)
        # apply actual encoder to packed input sequence
        # run entire batch through RNN

        enc_hiddens, (last_hidden, last_cell) = self.encoder(X_packed)
      
        # pad packed sequence of output
        enc_hiddens, enc_hiddens_length = pad_packed_sequence(enc_hiddens, batch_first=False)
        # permute s.t batch is leading dimenstion
        enc_hiddens = torch.permute(enc_hiddens, (1, 0, 2))

        ## form decoder initial hidden and cell states, last_hidden: [2, b, h], init_decoder_hidden: [b, 2h]
        init_decoder_hidden, init_decoder_cell = torch.concat((last_hidden[0], last_hidden[1]), 1), torch.concat((last_cell[0], last_cell[1]), 1)
        assert init_decoder_hidden.shape[0] == init_decoder_cell.shape[0] == source_padded.shape[1] and init_decoder_cell.shape[1] == init_decoder_hidden.shape[1] == 2*self.hidden_size
        # final linear transform from encoder last layer to decoder first layer (initial h and c states)
        init_decoder_hidden, init_decoder_cell = self.h_projection(init_decoder_hidden), self.c_projection(init_decoder_cell)
        dec_init_state = (init_decoder_hidden, init_decoder_cell)
        ### END YOUR CODE

        return enc_hiddens, dec_init_state
        


    def decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
               dec_init_state: Tuple[torch.Tensor, torch.Tensor], target_padded: torch.Tensor) -> torch.Tensor:
        """Compute combined output vectors for a batch.

        @param enc_hiddens (Tensor): Hidden states (b, src_len, h*2), where
                                     b = batch size, src_len = maximum source sentence length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks (b, src_len), where
                                     b = batch size, src_len = maximum source sentence length.
        @param dec_init_state (tuple(Tensor, Tensor)): Initial state and cell for decoder
        @param target_padded (Tensor): Gold-standard padded target sentences (tgt_len, b), where
                                       tgt_len = maximum target sentence length, b = batch size.

        @returns combined_outputs (Tensor): combined output tensor  (tgt_len, b,  h), where
                                        tgt_len = maximum target sentence length, b = batch_size,  h = hidden size
        """
        # Chop off the <END> token for max length sentences.
        target_padded = target_padded[:-1]

        # Initialize the decoder state (hidden and cell)
        dec_state = dec_init_state

        # Initialize previous combined output vector o_{t-1} as zero
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # Initialize a list we will use to collect the combined output o_t on each step
        combined_outputs = []

        ### YOUR CODE HERE (~9 Lines)
        ### TODO:
        ###     1. Apply the attention projection layer to `enc_hiddens` to obtain `enc_hiddens_proj`,
        ###         which should be shape (b, src_len, h),
        ###         where b = batch size, src_len = maximum source length, h = hidden size.
        ###         This is applying W_{attProj} to h^enc, as described in the PDF.
        ###     2. Construct tensor `Y` of target sentences with shape (tgt_len, b, e) using the target model embeddings.
        ###         where tgt_len = maximum target sentence length, b = batch size, e = embedding size.
        ###     3. Use the torch.split function to iterate over the time dimension of Y.
        ###         Within the loop, this will give you Y_t of shape (1, b, e) where b = batch size, e = embedding size.
        ###             - Squeeze Y_t into a tensor of dimension (b, e).
        ###             - Construct Ybar_t by concatenating Y_t with o_prev on their last dimension
        ###             - Use the step function to compute the the Decoder's next (cell, state) values
        ###               as well as the new combined output o_t.
        ###             - Append o_t to combined_outputs
        ###             - Update o_prev to the new o_t.
        ###     4. Use torch.stack to convert combined_outputs from a list length tgt_len of
        ###         tensors shape (b, h), to a single tensor shape (tgt_len, b, h)
        ###         where tgt_len = maximum target sentence length, b = batch size, h = hidden size.
        ###
        ### Note:
        ###    - When using the squeeze() function make sure to specify the dimension you want to squeeze
        ###      over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.
        ###
        ### You may find some of these functions useful:
        ###     Zeros Tensor:
        ###         https://pytorch.org/docs/stable/generated/torch.zeros.html
        ###     Tensor Splitting (iteration):
        ###         https://pytorch.org/docs/stable/generated/torch.split.html
        ###     Tensor Dimension Squeezing:
        ###         https://pytorch.org/docs/stable/generated/torch.squeeze.html
        ###     Tensor Concatenation:
        ###         https://pytorch.org/docs/stable/generated/torch.cat.html
        ###     Tensor Stacking:
        ###         https://pytorch.org/docs/stable/generated/torch.stack.html

        # apply attnention projection to h_enc
        enc_hiddens_proj = self.att_projection(enc_hiddens)
        src_len = enc_hiddens_proj.shape[1]
        assert enc_hiddens_proj.shape[0] == batch_size and enc_hiddens_proj.shape[1] == src_len and 2*enc_hiddens_proj.shape[2] == enc_hiddens.shape[2]

        # construct tensor `Y` of target sentences with shape (tgt_len, b, e) using the target model embeddings.
        # Y is tensor of actual embedding of predicted next word in traget vocab
        Y = self.model_embeddings.target(target_padded)
        all_split_Y = torch.split(Y, split_size_or_sections=1)
        assert len(all_split_Y) == target_padded.shape[0]
        # entire decode step in for loop
        for Y_t in all_split_Y:
            assert Y_t.shape[0] == 1 and Y_t.shape[1] == batch_size and Y_t.shape[2] == self.model_embeddings.embed_size
            Y_t = torch.squeeze(Y_t, dim=0)
            # combine current word embedding with previous o which is attention weighted encoder hiddenstates
            Ybar_t = torch.concat((Y_t, o_prev), dim=1)
            assert Ybar_t.shape[0] == batch_size and Ybar_t.shape[1] == self.model_embeddings.embed_size+self.hidden_size
            dec_state, o_t, e_t = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)
            assert len(dec_state) == 2
            assert o_t.shape[0] == batch_size and o_t.shape[1] == self.hidden_size
            combined_outputs.append(o_t)
            o_prev = o_t

        combined_outputs = torch.stack(combined_outputs)
        ### END YOUR CODE

        return combined_outputs


    ## calculate attention probability between decoder hidden states and all encoder hidden states, obtain attention weighted encoder hidden state
    ## concatenate with w. decoder hidden state and project to hidden dimension as combined output
    def step(self, Ybar_t: torch.Tensor,
             dec_state: Tuple[torch.Tensor, torch.Tensor],
             enc_hiddens: torch.Tensor,
             enc_hiddens_proj: torch.Tensor,
             enc_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """ Compute one forward step of the LSTM decoder, including the attention computation.

        @param Ybar_t (Tensor): Concatenated Tensor of [Y_t o_prev], with shape (b, e + h). The input for the decoder,
                                where b = batch size, e = embedding size, h = hidden size.
        @param dec_state (tuple(Tensor, Tensor)): Tuple of tensors both with shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's prev hidden state, second tensor is decoder's prev cell.
        @param enc_hiddens (Tensor): Encoder hidden states Tensor, with shape (b, src_len, h * 2), where b = batch size,
                                    src_len = maximum source length, h = hidden size.
        @param enc_hiddens_proj (Tensor): Encoder hidden states Tensor, projected from (h * 2) to h. Tensor is with shape (b, src_len, h),
                                    where b = batch size, src_len = maximum source length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks shape (b, src_len),
                                    where b = batch size, src_len is maximum source length.

        @returns dec_state (tuple (Tensor, Tensor)): Tuple of tensors both shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's new hidden state, second tensor is decoder's new cell.
        @returns combined_output (Tensor): Combined output Tensor at timestep t, shape (b, h), where b = batch size, h = hidden size.
        @returns e_t (Tensor): Tensor of shape (b, src_len). It is attention scores distribution.
                                Note: You will not use this outside of this function.
                                      We are simply returning this value so that we can sanity check
                                      your implementation.
        """

        combined_output = None

        ### YOUR CODE HERE (~3 Lines)
        ### TODO:
        ###     1. Apply the decoder to `Ybar_t` and `dec_state`to obtain the new dec_state.
        ###     2. Split dec_state into its two parts (dec_hidden, dec_cell)
        ###     3. Compute the attention scores e_t, a Tensor shape (b, src_len).
        ###        Note: b = batch_size, src_len = maximum source length, h = hidden size.
        ###
        ###       Hints:
        ###         - dec_hidden is shape (b, h) and corresponds to h^dec_t in the PDF (batched)
        ###         - enc_hiddens_proj is shape (b, src_len, h) and corresponds to W_{attProj} h^enc (batched).
        ###         - Use batched matrix multiplication (torch.bmm) to compute e_t (be careful about the input/ output shapes!)
        ###         - To get the tensors into the right shapes for bmm, you will need to do some squeezing and unsqueezing.
        ###         - When using the squeeze() function make sure to specify the dimension you want to squeeze
        ###             over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.
        ###
        ### Use the following docs to implement this functionality:
        ###     Batch Multiplication:
        ###         https://pytorch.org/docs/stable/generated/torch.bmm.html
        ###     Tensor Unsqueeze:
        ###         https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
        ###     Tensor Squeeze:
        ###         https://pytorch.org/docs/stable/generated/torch.squeeze.html

        batch_size, hidden_size = Ybar_t.shape[0], dec_state[0].shape[1]
        # apply decoder to obtain new dec_state, input is ybar_t, concatenated hidden state and attention output from encoder
        dec_state = self.decoder(Ybar_t, dec_state)
        # split the new state into hidden and cell
        dec_hidden, dec_cell = dec_state[0], dec_state[1]
        # calculate attention score
        # unsqueeze and expand new_dec_hidden into 3d
        dec_hidden_3d = torch.unsqueeze(dec_hidden, dim = -1)
        assert dec_hidden_3d.shape[0] == batch_size and dec_hidden_3d.shape[1] == hidden_size and dec_hidden_3d.shape[2] == 1 
        ## conduct batch matrix multiplication
        e_t = torch.bmm(enc_hiddens_proj, dec_hidden_3d)
        assert len(list(e_t.shape)) == 3 and e_t.shape[0] == enc_hiddens.shape[0] and e_t.shape[1] == enc_hiddens.shape[1] and e_t.shape[-1] == 1
        e_t = torch.squeeze(e_t, dim = -1)

        # Set e_t to -inf where enc_masks has 1
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))
        ### YOUR CODE HERE (~6 Lines)
        ### TODO:
        ###     1. Apply softmax to e_t to yield alpha_t
        ###     2. Use batched matrix multiplication between alpha_t and enc_hiddens to obtain the
        ###         attention output vector, a_t.
        ###           - alpha_t is shape (b, src_len)
        ###           - enc_hiddens is shape (b, src_len, 2h)
        ###           - a_t should be shape (b, 2h)
        ###           - You will need to do some squeezing and unsqueezing.
        ###     Note: b = batch size, src_len = maximum source length, h = hidden size.
        ###
        ###     3. Concatenate dec_hidden with a_t to compute tensor U_t
        ###     4. Apply the combined output projection layer to U_t to compute tensor V_t
        ###     5. Compute tensor O_t by first applying the Tanh function and then the dropout layer.
        ###
        ### Use the following docs to implement this functionality:
        ###     Softmax:
        ###         https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
        ###     Batch Multiplication:
        ###         https://pytorch.org/docs/stable/generated/torch.bmm.html
        ###     Tensor View:
        ###         https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
        ###     Tensor Concatenation:
        ###         https://pytorch.org/docs/stable/generated/torch.cat.html
        ###     Tanh:
        ###         https://pytorch.org/docs/stable/generated/torch.tanh.html

        # normalize e_t to get alpha_t
        alpha_t = torch.unsqueeze(F.softmax(e_t, dim=1), dim = 1)
        #print(alpha_t.shape, enc_hiddens.shape)
        a_t = torch.bmm(alpha_t, enc_hiddens)
        # bx1x2h
        a_t = torch.squeeze(a_t, dim = 1)
        assert a_t.shape[0] == batch_size and a_t.shape[1] == 2*hidden_size
        U_t = torch.cat((a_t, dec_hidden), 1)
        V_t = self.combined_output_projection(U_t)
        O_t = self.dropout(torch.tanh(V_t))
        combined_output = O_t
        
        return dec_state, combined_output, e_t



    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.

        @param enc_hiddens (Tensor): encodings of shape (b, src_len, 2*h), where b = batch size,
                                     src_len = max source length, h = hidden size.
        @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.

        @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                    where src_len = max source length, h = hidden size.
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)

    def beam_search(self, src_sent: List[str], beam_size: int = 5, max_decoding_time_step: int = 70) -> List[
        Hypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)

        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        src_encodings_att_linear = self.att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.tgt['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_t_embed = self.model_embeddings.target(y_tm1)

            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, _ = self.step(x, h_tm1,
                                                exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)

            # log probabilities over target words
            # this is where the actual prediction happens, beam search is used to predict next word
            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = torch.div(top_cand_hyp_pos, len(self.vocab.tgt), rounding_mode='floor')
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.model_embeddings.source.weight.device

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.model_embeddings.embed_size, hidden_size=self.hidden_size,
                         dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
