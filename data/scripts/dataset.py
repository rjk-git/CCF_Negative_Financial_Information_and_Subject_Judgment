from mxnet import gluon
import pandas as pd
from itertools import islice
from mxnet.gluon.data import ArrayDataset


class ClassDataset(gluon.data.Dataset):
    def __init__(self, file_path, **kwargs):
        super(ClassDataset, self).__init__(**kwargs)
        self.id, self.context, self.label = self._get_data(file_path)

    def _get_data(self, file_path):
        example_id = []
        example_content = []
        example_label = []
        data = pd.read_csv(file_path, doublequote=True)
        datas = data.values.tolist()
        for data in datas:
            eid = data[0]
            title = data[1]
            content = data[2]
            label = data[4]
            example_id.append(eid)
            if title != title:
                title = ""
            if content != content:
                content = ""
            example_content.append(title + content)
            example_label.append(label)
        return example_id, example_content, example_label

    def __getitem__(self, item):
        return self.id[item], self.context[item], self.label[item]

    def __len__(self):
        return len(self.id)


class ClassTestDataset(gluon.data.Dataset):
    def __init__(self, file_path, **kwargs):
        super(ClassTestDataset, self).__init__(**kwargs)
        self.id, self.context, self.example_entitys = self._get_data(file_path)

    def _get_data(self, file_path):
        example_id = []
        example_content = []
        example_entitys = []
        data = pd.read_csv(file_path, doublequote=True)
        datas = data.values.tolist()
        for data in datas:
            eid = data[0]
            title = data[1]
            content = data[2]
            entity = data[3]
            example_id.append(eid)
            if title != title:
                title = ""
            if content != content:
                content = ""
            if entity != entity:
                entity = ""
            example_content.append(title + content)
            example_entitys.append(entity)
        return example_id, example_content, example_entitys

    def __getitem__(self, item):
        return self.id[item], self.context[item], self.example_entitys[item]

    def __len__(self):
        return len(self.id)
