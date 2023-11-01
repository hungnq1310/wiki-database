import os
import logging
import torch

from configs.arguments import Arguments
from model import retriever_model
from database import make_database
from data import make_data
from database.make_database import create_milvus_connection
from pymilvus import utility, Collection
from sentence_transformers import SentenceTransformer

import dotenv

dotenv.load_dotenv()

DBNAME=os.getenv("DBNAME", "wiki_33m_milvus")
HOST=os.getenv("HOST", "localhost")
PORT=os.getenv("PORT", "19530")
USER=os.getenv("USER", "root")
PWD=os.getenv("PWD", "Milvus")
TB_WIKI=os.getenv("TB_WIKI", "wiki_tb")
TB_CLIENT=os.getenv("TB_CLIENT", "client_tb")
BATCH=int(os.getenv("BATCH", 200))

logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

def main():
    arguments = Arguments()
    args = arguments.parse()
    # row = 17553713 if args.dataset_version == "wiki40b_en_100_0" else 33849898
    row = 1000 if args.dataset_version == "wiki40b_en_100_0" else 1000
    # connect milvus server
    create_milvus_connection(host=HOST, port=PORT)
    # start to create
    if not args.just_create_index:
        collection = None
        if args.init_tb:
            if utility.has_collection(TB_WIKI):
                print("Drop collection")
                utility.drop_collection(TB_WIKI)
            print("Create collection")
            collection = make_database.create_wiki_table()
        elif not utility.has_collection(TB_WIKI): 
            raise ValueError("Not having table wiki, please initialize first with `--init_tb` arguements")
        else: 
            collection = Collection(TB_WIKI)  

        # model = SentenceTransformer('all-MiniLM-L6-v2')
        model = SentenceTransformer(args.model_name_pr_path)
        # specify target_devices for multi_gpus/gpu/cpu
        target_devices = ['cuda:{}'.format(i) for i in range(torch.cuda.device_count())]
        
        print("Start downloading dataset")
        wiki_snippets = make_data.download_dataset(
            dataset_name=args.dataset_name,
            dataset_version = args.dataset_version,
            streaming=args.streaming
        )

        print("Start inserting knowledges")
        make_database.using_sbert_encode_multi_gpu(
            collection=collection,
            model=model,
            snippets=wiki_snippets,
            target_devices=target_devices, 
            batch_insert=BATCH,
            limit_samples=row
        )
        make_database.build_indexs(collection=collection)
    else:
        logger.warning("Only index initialization is perfomed, make sure your table is filled up with data.")
        make_database.build_indexs(collection=collection)

if __name__=="__main__":
    main()


