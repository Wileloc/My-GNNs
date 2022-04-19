from statistics import mean

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from utils import set_random_seed