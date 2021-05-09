from torchtext.data import Filed, TabularDataset, BucketIterator


def tokenize(x): return x.split()


quote = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)
score = Field(sequential=False, use_vocab=False)

fields = {
    'quote': ('q', quote),
    'score': ('s', score)
}


train_data, test_data = TabularDataset.splits(
    path='mydata',
    train='train.json',
    test='test.json',
    format='json',
    fields=fields
)

quote.build_vocab(train_data,
                  max_size=10000,
                  min_freq=1)


train_iterator, test_iterator = BucketIterator.splits(
    (train_data,test_data),
    batch_size=2,
    device='cuda'
)


for batch in train_iterator:
    print(batch.q)
    print(batch.s)
    