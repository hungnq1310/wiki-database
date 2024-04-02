CUDA_VISIBLE_DEVICES=0 python src/run.py \
--tbname wiki_tb \
--model_name_or_path all-MiniLM-L6-v2 \
--just_create_index \
--index_type ivf_flat \
--metrics_type l2 \
--n_gpus 1 