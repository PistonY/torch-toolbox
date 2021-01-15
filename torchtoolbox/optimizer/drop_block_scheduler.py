from ..nn.norm import DropBlock2d


class DropBlockScheduler(object):
    def __init__(self, model, batches: int, num_epochs: int, start_value=0.1, stop_value=1.):
        self.model = model
        self.iter = 0
        self.start_value = start_value
        self.num_iter = batches * num_epochs
        self.st_line = (stop_value - start_value) / self.num_iter
        self.groups = []
        self.value = start_value

        def coll_dbs(md):
            if isinstance(md, DropBlock2d):
                self.groups.append(md)

        model.apply(coll_dbs)

    def update_values(self, value):
        for db in self.groups:
            db.p = value

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_value(self):
        self.value = self.st_line * self.iter + self.start_value

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if (key != 'model' and key != 'groups')}

    def step(self):
        self.get_value()
        self.update_values(self.value)
        self.iter += 1


class ObjectSchedule(object):
    def __init__(self, object, adjust_param, start_epoch, stop_epoch, batches, start_value, stop_value, mode='linear'):
        super().__init__()
        self.start_iter = start_epoch * batches
        self.end_iter = stop_epoch * batches
        self.start_value = start_value
        self.adjust_param = adjust_param
        self.object = object
        self.st_base = (stop_value - start_value) / \
            (self.end_iter - self.start_iter)
        self.iter = 0
        self._value = start_value

    def get_value(self):
        self._value = self.st_base * \
            (self.iter - self.start_iter) + self.start_value

    def update_value(self):
        setattr(self.object, self.adjust_param, self.value)

    def step(self):
        if not (self.iter < self.start_iter or self.iter > self.end_iter):
            self.get_value()
            self.update_value()
        self.iter += 1

    @property
    def value(self):
        return self._value

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'object'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)
