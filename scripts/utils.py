import logging
from nltk import bleu
import os
import csv


def config_logger(log_path):
    # Configuring logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fhandler = logging.FileHandler(log_path, mode='w')
    shandler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    shandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.addHandler(shandler)

    return logger


def get_entities(labels, content):
    entity = []
    entities = []
    for label, char in zip(labels, content):
        if label == "O":
            continue
        if label == "S":
            if len(entity) != 0:
                entities.append("".join(entity))
                entity = []
            entities.append(char)
        elif label == "B":
            if len(entity) != 0:
                entities.append("".join(entity))
            entity.append(char)
        elif label == "E":
            entities.append("".join(entity) + char)
            entity = []
        else:
            entity.append(char)
    if len(entity) != 0:
        entities.append("".join(entity))
    return entities


def ensmeble():
    results_file = os.listdir("../results")
    file_num = len(results_file) - 1
    ensmeble_guid_levels = {}
    for result_file in results_file:
        if result_file == "temp":
            continue
        guid_levels = open("../results/{}".format(result_file),
                           "r", encoding="utf-8").readlines()[1:]
        for guid_level in guid_levels:
            guid = guid_level.split(",")[0]
            level = guid_level.split(",")[1]
            if ensmeble_guid_levels.get(guid, 0) == 0:
                ensmeble_guid_levels.update({guid: int(level) / file_num})
            else:
                level = ensmeble_guid_levels.get(guid) + int(level) / file_num
                ensmeble_guid_levels[guid] = level
    f = open("../results/ensmeble_of_{}_files.csv".format(file_num), "w", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(["id", "label"])
    for k, v in ensmeble_guid_levels.items():
        guid = k
        level = round(v)
        row = [guid, level]
        writer.writerow(row)


if __name__ == "__main__":
    entities = get_entities(["O", "B", "E", "O", "O", "O", "E", "I", "I", "I", "E",
                             "O", "O", "O", "O", "I", "E", "O", "O", "O", "O", "O", "B"], "我今天来到了北京天安门，收到了京东发来的短信。")
    print(entities)
