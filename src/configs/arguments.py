import argparse
import logging

logger = logging.getLogger(__name__)

class Arguments:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                conflict_handler="resolve"
                )
        self.init_environment()
        self.init_dataset_args()

    def init_environment(self) -> None:
        """Provide environment variables
        """
        self.parser.add_argument(
            "--dbname",
            type=str,
            help="name of database to connect",
            default=""
        )
        self.parser.add_argument(
            "--n_gpus",
            type=int,
            help="number of availabel gpus",
            default=1
        )
        self.parser.add_argument(
            "--gpu_index",
            type=int,
            help="current gpu",
            default=0
        )
        self.parser.add_argument(
            "--host",
            type=str,
            help="host of database to connect",
            default=""
        )
        self.parser.add_argument(
            "--port",
            type=str,
            help="port of database to connect",
            default=""
        )
        self.parser.add_argument(
            "--user",
            type=str,
            help="user name to log in to database",
            default=""
        )
        self.parser.add_argument(
            "--pwd",
            type=str,
            help="password to log in to database",
            default=""
        )
        self.parser.add_argument(
            "--tbname",
            type=str,
            help="name of table in the database",
            default=""
        )
    def init_dataset_args(self):
        """Provide dataset information
        """

        self.parser.add_argument(
            "--init_db",
            action='store_true',
            help="whether to create a database or not",
        )
        self.parser.add_argument(
            "--just_create_index",
            action='store_true',
            help="You have data already inserted and just want to create index",
        )
        self.parser.add_argument(
            "--init_tb",
            action='store_true',
            help="whether to create a table or not",
        )
        self.parser.add_argument(
            "--dataset_name",
            type=str,
            help="name of dataset to be downloaded",
            default="wiki_snippets"
        )
        self.parser.add_argument(
            "--dataset_version",
            type=str,
            help="name of version of the dataset",
            default="wiki40b_en_100_0"
        )
        self.parser.add_argument(
            "--client_data_path",
            type=str,
            help="Knowledge file which is provided by client",
            default=""
        )
        self.parser.add_argument(
            "--streaming",
            type=bool,
            help="Whether tp use streaming mode or not",
            default=True
        )
        self.parser.add_argument(
            "--model_name_or_path",
            type=str,
            help="Specify model name to load Sbert Model",
            default=""
        )
        self.parser.add_argument(
            "--max_rows",
            type=int,
            help="maximun number of rows inserting to collection",
            default=None
        )
        self.parser.add_argument(
            "--metrics_type",
            type=str,
            help="type of metric when building index",
            choices=["l2", "ip", "cosine", "hamming", "jaccard"],
            default="l2"
        )
        self.parser.add_argument(
            "--index_type",
            type=str,
            help="type of index when building field's index",
            choices=['flat', 'ivf_flat', 'hnsw'],
            default="ivf_flat"
        )
        self.parser.add_argument(
            "--batch_embed",
            type=int,
            help="number of batch examples when model encode",
            default=32
        )
        self.parser.add_argument(
            "--batch_insert",
            type=int,
            help="number limit of batch examples to insert to milvus database",
            default=10000
        )
        


    def parse(self):
        """Get arguments
        """
        args = self.parser.parse_args()
        return args
