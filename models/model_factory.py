import logging
from models.Series2Vec import Series2Vec
from models.TS_TCC import TS_TCC
from models.TF_C import TF_C

####
logger = logging.getLogger('__main__')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def Model_factory(config, data):
    config['Data_shape'] = data['train_data'].shape
    config['num_labels'] = int(max(data['train_label'])) + 1

    if config['Model_Type'][0] == 'Series2Vec':
        model = Series2Vec.Seires2Vec(config, num_classes=config['num_labels'])
    if config['Model_Type'][0] == 'TS_TCC':
        model = TS_TCC.TS_TCC(config, num_classes=config['num_labels'])
    if config['Model_Type'][0] == 'TF_C':
        model = TF_C.TF_C(config, num_classes=config['num_labels'])

    logger.info("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(count_parameters(model)))
    return model
