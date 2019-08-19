# -*- coding: utf-8 -*-
# Author: pistonyang@gmail.com

from .loss import *
from .sequential import *
from .switch_norm import *
from .activation import *

try:
    from .parallel import *
except ImportError:
    pass
