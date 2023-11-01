from typing import (
        Tuple,
        Union,
        List
        )

from transformers import (
        DPRContextEncoderTokenizer,
        DPRContextEncoder
        )
import torch

def load_dpr_context_encoder(
        model_name_or_path: str
        )-> Tuple[DPRContextEncoder,DPRContextEncoderTokenizer]:
    """Load model DPR context encoder to encode knowledges

    Args:
        model_name_or_path: name of DPR model needs to be downloaded from
                            HuggingFace hub or path to DPR checkpoint
    """
    ctx_token = DPRContextEncoderTokenizer.from_pretrained(model_name_or_path)
    ctx_model = DPRContextEncoder.from_pretrained(model_name_or_path)

    return (ctx_model, ctx_token)

def get_ctx_embd(
        model_encoder: DPRContextEncoder,
        tokenizer: DPRContextEncoderTokenizer,
        text: Union[str, List[str]],
        device: torch.device
        ) -> torch.tensor:
    """Get knowledge embedding

    Args:
        model_encoder: DPR context encoder model
        tokenizer: DPR tokenizer
        text: a knowledge (sentence, paragraph,...)
    """
    model_encoder.eval()
    encoded_input = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        model_output = model_encoder(**encoded_input)
        batch_embeds = model_output["pooler_output"].cpu().detach().numpy()
        batch_embeds = [list(batch_embeds[i, :].reshape(-1).astype(float))
                            for i in range(len(batch_embeds))
                        ]
    return batch_embeds


def get_ctx_emb_sbert( 
        model_name_pr_path: str,
        batch_contents : Union[str, List[str]],
        target_devices: List[str] = None
        ) -> torch.tensor:
    """
    Using multiple processes (1 per GPU), which encode sentences in parallel. 
    This gives a near linear speed-up when encoding large text collections.
    """
    # handling batch_contents but only having one sentence
    if isinstance(batch_contents, str):
        batch_contents = [batch_contents]

    from sentence_transformers import SentenceTransformer
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    model = SentenceTransformer(model_name_pr_path)
    # Start the multi-process pool on all available CUDA devices
    pool = model.start_multi_process_pool(target_devices=target_devices)

    # Compute the embeddings using the multi-process pool
    batch_embeds = model.encode_multi_process(batch_contents, pool)
    print("Embeddings computed. Shape:", batch_embeds.shape)

    #O ptional: Stop the proccesses in the pool
    model.stop_multi_process_pool(pool)

    return batch_embeds

