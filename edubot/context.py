
from time import time as current_time_seconds
from dataclasses import dataclass, field
from abc import ABC
from typing import Text
from collections import defaultdict


class ContextStorage(ABC):
    def store_utterance(self, utterance: Text, conv_id: Text):
        raise NotImplementedError

    def retrieve_context(self, conv_id: Text):
        raise NotImplementedError


@dataclass
class InMemoryEntry:
    context: list = field(default_factory=lambda: [])
    time_created: float = field(default_factory=lambda: current_time_seconds())


class InMemoryContextStorage(ContextStorage):
    def __init__(self):
        self.context_cache = defaultdict(InMemoryEntry)

    def store_utterance(self, utterance: Text, conv_id: Text):
        self.context_cache[conv_id].context.append(utterance)

    def retrieve_context(self, conv_id: Text):
        return self.context_cache[conv_id].context

    def clear_cache(self):
        current_time = current_time_seconds()
        to_be_deleted = [conv_id for conv_id, entry in self.context_cache.items()
                         if current_time - entry.time_created > 7200]
        for id_to_delete in to_be_deleted:
            del self.context_cache[id_to_delete]
