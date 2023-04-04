import os
import csv
import sys
import tqdm
import argparse
import pandas as pd

sys.path.append("..")
from argparse import Namespace
from fairseq.data import encoders
from fairseq.data.audio.speech_to_text_dataset import S2TDataConfig

parser = argparse.ArgumentParser()
parser.add_argument("--lang", help="target language")
args = parser.parse_args()

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

splits = ['dev', 'tst-COMMON', 'tst-HE', 'train']
data_dir = os.path.join(root, 'data', 'mustc', f'en-{args.lang}')

config_file = os.path.join(data_dir, "config_raw.yaml")
data_cfg = S2TDataConfig(config_file)
dict_path = os.path.join(data_dir, data_cfg.vocab_filename)
pre_tokenizer = encoders.build_tokenizer(Namespace(**data_cfg.pre_tokenizer))
bpe_tokenizer = encoders.build_bpe(Namespace(**data_cfg.bpe_tokenizer))

def save_df_to_tsv(dataframe, path):
    dataframe.to_csv(
        path,
        sep="\t",
        header=True,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
    )

def load_df_from_tsv(path):
    return pd.read_csv(
        path,
        sep="\t",
        header=0,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
        na_filter=False,
    )

def conv_calc(input_len, conv_layers):
    output_len = input_len
    for kernel, stride, padding in conv_layers:
        output_len = int((output_len + 2 * padding - (kernel - 1) - 1) / stride + 1)
    return output_len

def process_audio(word_time, n_frames):
    word_time = list(map(float, word_time.split(",")))
    # wav2vec卷积与额外的2层CNN卷积
    # 1D卷积: (kernel_size, stride, padding)
    conv_layers = [
        (10, 5, 0),
        (3, 2, 0),
        (3, 2, 0),
        (3, 2, 0),
        (3, 2, 0),
        (2, 2, 0),
        (2, 2, 0),
        (5, 2, 2),    # 长度缩小一半
        (5, 2, 2),    # 长度再次缩小一半
    ]
    # 计算最终这句语音经过卷积之后的维度
    n_hidden = conv_calc(n_frames, conv_layers)
    L, R = [], []
    L.append(0)
    total_time = word_time[-1]
    for i in range(len(word_time)):
        # 从这里判断应该是结束时间
        R.append(int(word_time[i] / total_time * n_hidden + 0.5) - 1)
        if i < len(word_time) - 1:
            L.append(R[-1] + 1)   # 前一帧+1
    # 开始帧，结束帧(相对偏移!!!)
    result = [f"{L[i]},{R[i]}" for i in range(len(word_time))]
    result = '|'.join(result)
    return result

def process_text(word_text, src_text):
    # 处理子词化后对应的位置
    word_text = word_text.split(",")
    if pre_tokenizer is not None:
        src_text = pre_tokenizer.encode(src_text)
    if bpe_tokenizer is not None:
        src_text = bpe_tokenizer.encode(src_text)
    src_text = src_text.split(" ")
    L, R = [], []
    cur = 0
    for i in range(len(word_text)):
        if word_text[i] == "":
            if i == 0:
                L.append(1)
                R.append(0)
            else:
                L.append(R[i - 1] + 1)
                R.append(L[i] - 1)
        else:
            L.append(cur)
            cur = cur + 1
            while cur < len(src_text) and src_text[cur][0] != "▁":
                cur = cur + 1
            R.append(cur - 1)
    result = [f"{L[i]},{R[i]}" for i in range(len(word_text))]
    result = '|'.join(result)
    return result

def main():
    for split in splits:
        tsv_file = os.path.join(data_dir, split + "_raw_seg.tsv")
        df = load_df_from_tsv(tsv_file)
        data = list(df.T.to_dict().values())
        pbar = tqdm.tqdm(range(len(data)))
        for item in data:
            pbar.update()
            # NOTE: word_time与word_text之间是严格对应的
            # word-time与n_frames之间的对齐
            # 语音与经过CNN卷积后的帧数的对齐
            item["audio_align"] = process_audio(item["word_time"], item["n_frames"])
            # word-text与src_text之间的对齐
            # 语音对应的文本(某些片段可能没有单词)与transcript之间的对齐
            # bpe之后如何知道是语音对应的文本
            item["text_align"] = process_text(item["word_text"], item["src_text"])
        df = pd.DataFrame.from_dict(data)
        save_df_to_tsv(df, os.path.join(data_dir, split + "_raw_seg_plus.tsv"))

if __name__ == "__main__":
    main()