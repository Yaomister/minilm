import torch


class DataLoader:
    """The dataloader."""

    def __init__(self, tokens, batch_size, sequence_length, rank, world_size=1):
        # the vocab size
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        # the batch size to process per training step
        self.batch_size = batch_size
        # the number of tokens the model can handle at once
        self.sequence_length = sequence_length

        # DDP stuff for feeding data to models running on different nodes
        self.rank = rank
        self.world_size = world_size
        self.start = rank * batch_size * sequence_length
        self.current = self.start

    def next_batch(self):
        """Get the next batch of data."""
        length = self.batch_size * self.sequence_length + 1
        buffer = self.tokens[self.current : self.current + length]

        # make sure we dont exceed the amount of data available
        if len(buffer) < length:
            self.current = self.start
            buffer = self.tokens[self.current : self.current + length]

        x = buffer[:-1].view(self.batch_size, self.sequence_length)

        # the y is always one off from x because we're predicting the next token
        y = buffer[1:].view(self.batch_size, self.sequence_length)

        # increase the index for when we're grabbing the next sequence
        self.current += (length - 1) * self.world_size

        return x, y
