import argparse
import logging
import math
import os
import sys

import numpy as np
import torchaudio
from tqdm import tqdm


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_hubert_feature")


def single_job(path):
    wav, sr = torchaudio.load(path)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    input_values = feature_extractor(wav.squeeze(0), sampling_rate=sample_rate, return_tensors="pt").input_values
    hubert_output = hubert_model(input_values.to(hubert_model.device), output_hidden_states=True)
    hubert_rep = hubert_output.last_hidden_state
    return hubert_rep


def main(tsv_dir, split, nshard, rank, feat_dir):
    generator, num = get_path_iterator(f"{tsv_dir}/{split}.lst", nshard, rank)
    iterator = generator()

    for path in tqdm(iterator):
        embedding = single_job(path)
        name = os.path.splitext(os.path.join(*(path.split("/")[-4:])))[0]
        feat_path = f"{feat_dir}/{name}.npy"
        os.makedirs(os.path.dirname(feat_path), exist_ok=True)

        embedding = embedding.squeeze(0)
        assert len(embedding.shape) == 2
        np.save(feat_path, embedding.detach().cpu().numpy())
        del embedding


def get_path_iterator(lst, nshard, rank):
    root = "/mnt/lynx2/datasets/SpeechLLM/LibriTTS/raw"
    with open(lst, "r") as f:
        lines = [line.rstrip() for line in f]
        tot = len(lines)
        shard_size = math.ceil(tot / nshard)
        start, end = rank * shard_size, min((rank + 1) * shard_size, tot)
        assert start < end, "start={start}, end={end}"
        logger.info(f"rank {rank} of {nshard}, process {end - start} ({start}-{end}) out of {tot}")

        lines = lines[start:end]

        def iterate():
            for line in lines:
                yield os.path.join(root, line)

        return iterate, len(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_dir")
    parser.add_argument("split")
    parser.add_argument("nshard", type=int)
    parser.add_argument("rank", type=int)
    parser.add_argument("feat_dir")
    args = parser.parse_args()

    semantic_model_path = "facebook/hubert-large-ll60k"
    from transformers import AutoFeatureExtractor, AutoModel

    feature_extractor = AutoFeatureExtractor.from_pretrained(semantic_model_path)
    hubert_model = AutoModel.from_pretrained(semantic_model_path).eval().cuda()
    sample_rate = 16000
    kwargs = vars(args)
    main(**kwargs)
