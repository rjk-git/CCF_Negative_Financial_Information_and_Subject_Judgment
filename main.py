import pandas as pd


class PickKeyEntity(object):
    # neg_filepath: your result of negative predict, only negative 0-1
    # train_filepath: source "Train_Data.csv"
    # test_filepath: source "Test_Data.csv"
    def __init__(self, neg_filepath, train_filepath, test_filepath):
        self.train_filepath = train_filepath
        self.test_filepath = test_filepath
        self.neg_filepath = neg_filepath
        self.ratio = 0.1  # in entity and in key_entitiy ratio that smaller than this ratio is not key entity
        self.nike = self._generate_nike_data()

    def run(self):
        test_data = pd.read_csv(self.test_filepath)
        test_data = test_data.loc[:, ["id", "entity"]]
        neg_data = pd.read_csv(self.neg_filepath)
        test_data = pd.merge(neg_data, test_data, on="id", how="inner")
        test_data["key_entity"] = test_data.apply(self.func_on_row, axis=1)
        test_data.to_csv("./result.csv", index=False,
                         columns=["id", "negative", "key_entity"])

    # add your thoughts here to filter entities if you want to improve the effect further.
    def func_on_row(self, row):
        key_entitys = []
        if row["negative"] == 1:
            entitys = row["entity"].split(";")
            for entity in entitys:
                if entity not in self.nike:
                    key_entitys.append(entity)
            key_entitys = self._remove_substring(key_entitys)
        return ";".join(key_entitys)

    # 'nike' entity means "not in key entitiy" but in entity from train data
    def _generate_nike_data(self):
        nike = []
        numsOfEntitiyAsKey = {}
        train_data = pd.read_csv(self.train_filepath)
        train_data = train_data.loc[:, ["negative", "entity", "key_entity"]]
        train_data = train_data[train_data.negative == 1]
        for index, row in train_data.iterrows():
            entitys = row["entity"]
            key_entitys = row["key_entity"]
            entitys = entitys.split(";")
            key_entitys = key_entitys.split(";")
            for entity in entitys:
                if numsOfEntitiyAsKey.get(entity, -1) == -1:
                    if entity in key_entitys:
                        numsOfEntitiyAsKey.update({entity: {"in": 1, "out": 0}})
                    else:
                        numsOfEntitiyAsKey.update({entity: {"in": 0, "out": 1}})
                else:
                    if entity in key_entitys:
                        numsOfEntitiyAsKey[entity]["in"] += 1
                    else:
                        numsOfEntitiyAsKey[entity]["out"] += 1
        for entity, nums in numsOfEntitiyAsKey.items():
            num_in = nums["in"]
            num_out = nums["out"]
            freq_in = num_in / (num_in + num_out)
            if freq_in < self.ratio:
                nike.append(entity)
        return nike

    # remove entities that can be substring of other entities.
    # eg: 资易贷，小资易贷，资易贷有限公司 we retain 小资易贷，资易贷有限公司
    def _remove_substring(self, entities):
        entities = list(set(entities))
        longest_entities = []
        for entity in entities:
            flag = 0
            for entity_ in entities:
                if entity == entity_:
                    continue
                if entity_.find(entity) != -1:
                    flag = 1
            if flag == 0:
                longest_entities.append(entity)
        return longest_entities


if __name__ == "__main__":
    pickKeyEntity = PickKeyEntity("result_negative_0.38824.csv", "Train_Data.csv", "Test_Data.csv")
    pickKeyEntity.run()
