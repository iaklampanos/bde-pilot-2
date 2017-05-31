import numpy as np
from sklearn.preprocessing import normalize


class Dataset(object):

    _items = None
    _items_iterator = None
    _dims = None
    _similarities = None

    def __init__(self, items, items_iterator, dims, similarities=None):
        self._items = items
        self._dims = dims
        self._items_iterator = items_iterator
        self._similarities = similarities

    def get_items(self):
        return self._items

    def get_items_iterator(self):
        return self._items_iterator

    def get_dims(self):
        return self._dims

    def get_similarities(self):
        return self._similarities

    def get_next(self):
        it_length = len(self._items)
        it_range = range(0, it_length)
        return it_range[0::self.get_items_iterator]
