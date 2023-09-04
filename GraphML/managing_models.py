import sqlite3
import config
from SQLHelpers.query_helpers import *
from Model.model_wrapper import ModelWrapper


class RegistrationDesk:
    @staticmethod
    def register_a_model(m_wrapp: ModelWrapper):
        conf = m_wrapp.storage.get_information_for_registration()
        dir_ = conf['dir']
        view_name = conf['view_name']

        con = sqlite3.connect(config.SQLite3_alg_db_path)
        cur = con.cursor()
        cur.execute("insert into {0}(dir, model) values({1},{2})".format(
            config.models_nav_table_name,
            SqlString(dir_), SqlString(view_name)
        ))
        con.commit()
        cur.close()
        con.close()
