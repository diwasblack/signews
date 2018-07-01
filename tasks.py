from invoke import task

from crits.dataset import CriticalTextDataset


@task
def save_data(c, doc=False):
    dataset = CriticalTextDataset()
    dataset.save_data()


@task
def dump_data(c, doc=False):
    dataset = CriticalTextDataset()
    dataset.dump_data()
