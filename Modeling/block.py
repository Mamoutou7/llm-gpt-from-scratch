class Block(nn.module):

  def __init__(self, num_embed, num_head):
    super().__init__()
    head_size = num_embed // num_head
    self.self_attention = MultiheadAttention(num_head, head_size)
    self.feed_forward = FeedForward(num_embed)
    self.linear_1 = nn.LayerNorm(num_embed)
    self.linear_2 = nn.LayerNorm(num_embed)


def forward(self, x):

  y = self.self_attention(x)
  x = self.linear_1(x + y)
  y = self.feed_forward(x + y)
  x = self.linear_2(x + y)

  return x