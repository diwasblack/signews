from invoke import task

from signews.dataset import SignificantTextDataset


@task
def save_data(c, doc=False):
    dataset = SignificantTextDataset()
    dataset.save_data()


@task
def dump_data(c, doc=False):
    dataset = SignificantTextDataset()
    dataset.dump_data()
