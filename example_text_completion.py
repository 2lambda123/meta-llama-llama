# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List

import fire

from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
    """Entry point of the program for generating text using a pretrained model.

    :param ckpt_dir: The directory containing checkpoint files for the pretrained model.
    :type ckpt_dir: str
    :param tokenizer_path: The path to the tokenizer model used for text encoding/decoding.
    :type tokenizer_path: str
    :param temperature: The temperature value for controlling randomness in generation.
            Defaults to 0.6.
    :type temperature: float
    :param top_p: The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
    :type top_p: float
    :param max_seq_len: The maximum sequence length for input prompts. Defaults to 128.
    :type max_seq_len: int
    :param max_gen_len: The maximum length of generated sequences. Defaults to 64.
    :type max_gen_len: int
    :param max_batch_size: The maximum batch size for generating sequences. Defaults to 4.
    :type max_batch_size: int
    :param ckpt_dir: str: 
    :param tokenizer_path: str: 
    :param temperature: float:  (Default value = 0.6)
    :param top_p: float:  (Default value = 0.9)
    :param max_seq_len: int:  (Default value = 128)
    :param max_gen_len: int:  (Default value = 64)
    :param max_batch_size: int:  (Default value = 4)

    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    prompts: List[str] = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        """A brief message congratulating the team on the launch:

        Hi everyone,
        
        I just """,
        # Few shot prompt (providing a few examples before asking model to complete more);
        """Translate English to French:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
    ]
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
