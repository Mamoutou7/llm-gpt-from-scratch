class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        self.position_embedding_table = nn.Embedding(block_size, num_embed)
        self.blocks = nn.Sequential(*[torch.Block(num_embed, n_head=num_head) for _ in range(num_layer)])
        self.linear_f = nn.LayerNorm(num_embed) # final layer norm
        self.lang_model_head = nn.Linear(num_embed, vocab_size)

    def _init_weights(self, module):
      if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
          torch.nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, index, targets=None):
        logits = self.token_embedding_table(index)

        token_embedding = self.token_embedding_table(idx)
        position_emb = self.position_embedding_table(torch.arange(T, device=device))
        # boradcating
        x = token_embedding + position_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.linear_f(x)  # (B,T,C)
        logits = self.lang_model_head(x) # (B,T,C, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self.forward(index)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1) # (B, T+1)
        return index



model = BigramLanguageModel(vocab_size)
m = model.to(device)

context = torch.zeros((1,1), dtype=torch.long, device=device)
generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist())
print(generated_chars)