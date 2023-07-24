# python evaluate_mmlu.py -m /path/to/Baichuan-7B

import argparse
import os
import sys
sys.path.append("..")
import torch
import numpy as np
import pandas as pd
from categories import subcategories, categories
from transformers import AutoTokenizer,AutoModelForCausalLM,LlamaForCausalLM, LlamaTokenizer
from mypeft import PeftModel
import time
choices = ["A", "B", "C", "D"]

def estimate_accs_of_bins_ranked_by_unc(is_correct, uncertainties, num_bins=10):
    is_correct = torch.Tensor(is_correct)
    uncertainties = torch.Tensor(uncertainties)
    is_correct_ = is_correct[torch.argsort(uncertainties)]
    is_correct_ = is_correct_[:is_correct_.shape[0] // num_bins * num_bins].view(num_bins, -1)
    return is_correct_.mean(-1) * 100.

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval(args, subject, model, tokenizer, dev_df, test_df):
    # print(model)
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]
    uncertainties = []

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        print(prompt_end)
        print("*****************************")
        # exit()
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        # print(prompt)
        # print("***************************")
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        while input_ids.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        label = test_df.iloc[i, test_df.shape[1] - 1]
        print("label",label)


        is_evaluation = True
        output, uncertainty = model(
            input_ids=input_ids,
            is_generation_mode=True,
        )
        print("output.logits.shape",output.logits.shape)
        logits = output.logits[:,-1].flatten()
        # print("uncertainty.shape",uncertainty.shape)
        # print("uncertainty",uncertainty)
        uncertainties.append(uncertainty.item())

        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer("A").input_ids[-1]],
                        logits[tokenizer("B").input_ids[-1]],
                        logits[tokenizer("C").input_ids[-1]],
                        logits[tokenizer("D").input_ids[-1]],
                    ]
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .to(torch.float32)
            .numpy()
        )
        print("probs",probs)
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)
        exit()

    acc = np.mean(cors)
    # print(cors)
    # print(uncertainties)
    accs_of_bins_ranked_by_unc = estimate_accs_of_bins_ranked_by_unc(cors, uncertainties)
    print(accs_of_bins_ranked_by_unc)
    cors = np.array(cors)
    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))
    # exit()

    return cors, acc, all_probs

"""
tokenizer = LlamaTokenizer.from_pretrained(base_model)
model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
"""


def main(args):

    tokenizer = LlamaTokenizer.from_pretrained(args.model,use_fast=False,add_bos_token=False, model_max_length=4096,padding_side="right")
    model = LlamaForCausalLM.from_pretrained(args.model,  torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(
            model,
            args.lora_weight,
            torch_dtype=torch.bfloat16,
        )
    model.eval()
    print("have_loaded_model")
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(args.model))):
        os.makedirs(os.path.join(args.save_dir, "results_{}".format(args.model)))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs = eval(args, subject, model, tokenizer, dev_df, test_df)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(args.model)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(args.model, choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(
                args.save_dir, "results_{}".format(args.model), "{}.csv".format(subject)
            ),
            index=None,
        )

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--ngpu", "-g", type=int, default=8)
    parser.add_argument("--data_dir", "-d", type=str, default="/home/myb/lora_llama/data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--lora_weight", "-l",type=str,  default="/home/myb/pyllama_data/yhamam1v2_0710") #yhamam1v5_0716_r16 #yhamam1v2_0710
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="yahma/llama-7b-hf",
    )
    args = parser.parse_args()
    main(args)
