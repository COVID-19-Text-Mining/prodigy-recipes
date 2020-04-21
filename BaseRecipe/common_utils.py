import json
import pymongo

###########################################
# communicate with mongodb
###########################################

def get_mongo_db(config_file_path):
    """
    read config file and return mongo db object

    :param config_file_path: path to config file. config is a dict of dict as
                config = {
                'mongo_db': {
                    'host': 'mongodb05.nersc.gov',
                    'port': 27017,
                    'db_name': 'COVID-19-text-mining',
                    'username': '',
                    'password': '',
                }
            }
    :return:
    """
    with open(config_file_path, 'r') as fr:
        config = json.load(fr)

    client = pymongo.MongoClient(
        host=config['mongo_db']['host'],
        port=config['mongo_db']['port'],
        username=config['mongo_db']['username'],
        password=config['mongo_db']['password'],
        authSource=config['mongo_db']['db_name'],
    )
    db = client[config['mongo_db']['db_name']]
    return db