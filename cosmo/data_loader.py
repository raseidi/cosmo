import torch
from torch.utils.data import Dataset, DataLoader


class EventLog(Dataset):
    def __init__(self, events, condition, ngram_indexes) -> None:
        super().__init__()
        self.events = events
        self.condition = condition
        self.ngram_indexes = ngram_indexes

    def __len__(self):
        return len(self.ngram_indexes) - 1  # there's no next activity for the last row

    def __getitem__(self, index):
        return (
            (
                self.events[self.ngram_indexes[index]],  # ngram
                self.condition[self.ngram_indexes[index]][-1],  # condition
            ),
            (
                self.events[self.ngram_indexes[index + 1]][-1, 0],  # next act
                self.events[self.ngram_indexes[index + 1]][-1, 1],  # next res
                self.events[self.ngram_indexes[index + 1]][-1, 2],  # next rt
            ),
        )


def get_loader(data, batch_size=64, shuffle=True):
    events = torch.tensor(data[0], dtype=torch.float)
    condition = torch.tensor(data[1], dtype=torch.long)
    cases_ngrams = torch.tensor(data[2], dtype=torch.long)

    event_dataset = EventLog(events, condition, cases_ngrams)
    log_loader = DataLoader(event_dataset, batch_size=batch_size, shuffle=shuffle)
    return log_loader
