import torch
import numpy as np
import pandas as pd
from generate_attacked import load_model_and_tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =", device)


def get_batches(texts, batch_size):
    return [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]


def compute_logits(batch, tokenizer, model):
    """
    This function computes the logits for a batch of sentences.

    Args:
        batch: a list of sentences
        tokenizer: the tokenizer to use
        model: the model to use

    Returns:
        The logits for the batch of sentences
    """
    if not isinstance(batch[0], list):
        batch = [batch]
    # print("batch =", batch)

    batch = batch[0]
    input = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(
        device
    )
    with torch.no_grad():
        model.to(device)
        # print("model device =", next(model.parameters()).is_cuda)
        output = model(**input)
        del input
        torch.cuda.empty_cache()
        return output.logits.cpu().numpy()


def compute_logits_difference(
    sentence, logits, model, tokenizer, max_sentence_size=512
):
    """
    This function computes the difference between the original prediction logit and the highest logit for each word.

    Args:
        sentence: the sentence to compute the logits difference for (string)
        logits: the logits for the sentence (numpy array)
        model: the model to use
        tokenizer: the tokenizer to use
        max_sentence_size: the maximum number of words to consider in the sentence (default: 512)

    Returns:
        The logits difference for each word in the sentence (numpy array)
    """
    # Remove the special tokens from the sentence
    sentence = sentence.replace("[[", "")
    sentence = sentence.replace("]]", "")

    n_classes = len(logits)
    predicted_class = np.argmax(
        logits
    )  # Predicted class for whole sentence using previously computed logits
    original_class_logit = logits[0][
        predicted_class
    ]  # Store the original prediction logit

    truncated_sentence = sentence.split(" ")[
        :max_sentence_size
    ]  # The tokenizer will only consider 512 words so we avoid computing unnecessary logits

    modified_sentences = []

    # Here, we replace each word by [UNK] and generate all sentences to consider
    for i, word in enumerate(truncated_sentence):
        modified_sentence = truncated_sentence.copy()
        modified_sentence[i] = "[UNK]"
        modified_sentence = " ".join(modified_sentence)
        modified_sentences.append(modified_sentence)

    # We cannot run more than 350 predictions simultaneously because of resources.
    # Split in batches if necessary.
    # Compute logits for all replacements.
    all_logits = []
    batches = [
        modified_sentences[i : i + 200] for i in range(0, len(modified_sentences), 200)
    ]
    for batch in batches:
        batch_logits = compute_logits(batch, tokenizer, model)
        all_logits.append(batch_logits)

    sentence_logits = np.concatenate(all_logits)

    # Get highest logit for each word excluding the original prediction
    arr_excluded = np.delete(sentence_logits, predicted_class, axis=1)

    # Calculate the maximum in each row excluding the original prediction
    max_values = np.max(arr_excluded, axis=1)

    # Compute the difference between the correct prediction and the highest other logit for each word
    logits_diff = sentence_logits[:, predicted_class] - max_values

    # Sort the logits difference in ascending order
    sorted_indices = np.argsort(logits_diff)

    return logits_diff[sorted_indices]


def compute_logits_difference_padding(
    sentence, logits, model, tokenizer, target_size=512
):
    """
    This function provides a wrapper for compute_logits_difference and includes padding to computations.
    """
    data = compute_logits_difference(sentence, logits, model, tokenizer, target_size)
    data = data.reshape(-1, 1)

    data_size = min(512, data.shape[0])
    # target = torch.zeros(target_size, 1).to(device)
    target = np.zeros((target_size, 1))
    target[:data_size, :] = data

    return target


def process_csv(
    input_csv, output_csv, model, tokenizer, target_size=512, adversarial=False
):
    """
    This function processes the input CSV file and computes the logits differences for each example.

    The data is stored in the output CSV file.

    This should be used if the CSV file was created via the TextAttack log_to_csv function.
    """

    # Read the input CSV file
    df = pd.read_csv(input_csv)

    # Calculate the logit differences for each row in the DataFrame
    logit_diffs = []
    for index, row in df.iterrows():
        print("Processing row {}".format(index))
        original_text = row["original_text"]
        perturbed_text = row["perturbed_text"]
        original_score = row["original_score"]
        perturbed_score = row["perturbed_score"]
        original_output = row["original_output"]
        perturbed_output = row["perturbed_output"]
        ground_truth_output = row["ground_truth_output"]
        num_queries = row["num_queries"]
        result_type = row["result_type"]

        text = original_text if not adversarial else perturbed_text
        text = text.replace("[[", "")
        text = text.replace("]]", "")
        # print("Text: ", text)
        logits = compute_logits([text], tokenizer, model)
        # Compute logits for the original_text using your model and tokenizer

        # Calculate the logit differences using the `compute_logits_difference_padding` function
        logits_diff = compute_logits_difference_padding(
            text,
            logits,
            model,
            tokenizer,
            target_size,
        )
        logit_str = logits_diff.flatten()
        logit_str = np.array2string(logit_str, separator=";")

        logit_diffs.append(logit_str)

    # Create a new df
    df_logit_diffs = pd.DataFrame()

    # Add the logit differences as a new column to the DataFrame
    df_logit_diffs["logit_diffs"] = logit_diffs

    # Add column indicating whether the example is adversarial or not (as defined by the adversarial parameter)
    df_logit_diffs["is_adversarial"] = adversarial

    df_logit_diffs["is_adversarial_success"] = (
        False if not adversarial else df["original_output"] != df["perturbed_output"]
    )

    if adversarial:
        df_logit_diffs["text"] = df["perturbed_text"]
        df_logit_diffs["confidence"] = df["perturbed_score"]
    else:
        df_logit_diffs["text"] = df["original_text"]
        df_logit_diffs["confidence"] = 1 - df["original_score"]

    # Write the modified DataFrame to a new CSV file
    df_logit_diffs.to_csv(output_csv, index=False)

    return df_logit_diffs


def process_csv2(
    input_csv, output_csv, model, tokenizer, target_size=512, adversarial=True
):
    """Use process_csv2 on a file that was created with logging everything"""
    # Read the input CSV file
    df = pd.read_csv(input_csv)

    # Calculate the logit differences for each row in the DataFrame
    logit_diffs = []
    for index, row in df.iterrows():
        print("Processing row {}".format(index))
        text = row["attacked_text"]

        # text = original_text if not adversarial else perturbed_text
        text = text.replace("[[", "")
        text = text.replace("]]", "")

        logits = compute_logits([text], tokenizer, model)
        # Compute logits for the original_text using your model and tokenizer

        # Calculate the logit differences using the `compute_logits_difference_padding` function
        logits_diff = compute_logits_difference_padding(
            text,
            logits,
            model,
            tokenizer,
            target_size,
        )
        logit_str = logits_diff.flatten()
        logit_str = np.array2string(logit_str, separator=";")

        logit_diffs.append(logit_str)

    # Create a new df
    df_logit_diffs = pd.DataFrame()

    # Add the logit differences as a new column to the DataFrame
    df_logit_diffs["logit_diffs"] = logit_diffs

    # Write the modified DataFrame to a new CSV file
    df_logit_diffs.to_csv(output_csv, index=False)

    return df_logit_diffs


if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer("sst-2")
    model.to(device)

    process_csv2(
        "all_log_sst2.csv",
        "sst2_logits.csv",
        model,
        tokenizer,
        adversarial=True,
    )
