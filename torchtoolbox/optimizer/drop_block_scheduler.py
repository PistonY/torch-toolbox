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
        return {
            key: value for key,
            value in self.__dict__.items() if (key != 'model' and key != 'groups')}

    def step(self):
        self.get_value()
        self.update_values(self.value)
        self.iter += 1
