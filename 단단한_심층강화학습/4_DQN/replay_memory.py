from collections import namedtuple, deque
import random
import itertools


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    def __init__(self, capacity):
        # deque는 양방향 queue를 의미한다.
        self.memory = deque([], maxlen=capacity)
        self.recent_point = 0
        self.max_capacity = capacity
        
        
    def push(self, *args):
        # Transition을 저장하는 부분이다.
        self.memory.append(Transition(*args))            


    def sample(self, batch_size=32):
        # memory로부터 batch_size 길이 만큼의 list를 반환한다.
        return random.sample(self.memory, batch_size)


    def slice2end(self, start):
        return list(itertools.islice(self.memory, start, len(self)+1))

    def __len__(self):
        # memory의 길이를 반환한다.
        return len(self.memory)
    