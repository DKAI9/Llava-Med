import torch

from models.llava_med_trainable import left_pad_1d_tensors, slice_generated_tokens


def test_left_pad_1d_tensors():
    seq_a = torch.tensor([101, 102, 103], dtype=torch.long)
    seq_b = torch.tensor([201, 202], dtype=torch.long)
    input_ids, attention_mask = left_pad_1d_tensors([seq_a, seq_b], pad_token_id=0)

    assert input_ids.tolist() == [[101, 102, 103], [0, 201, 202]]
    assert attention_mask.tolist() == [[1, 1, 1], [0, 1, 1]]


def test_slice_generated_tokens_from_left_padded():
    input_len = 4
    sequences = torch.tensor(
        [
            [9, 10, 11, 12, 101, 102],
            [0, 0, 21, 22, 201, 202],
        ],
        dtype=torch.long,
    )
    gen_ids = slice_generated_tokens(sequences, input_len)

    assert gen_ids.tolist() == [[101, 102], [201, 202]]
