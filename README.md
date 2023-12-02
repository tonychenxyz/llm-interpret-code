# Requirements
    transformers==4.34.0
# Changing Transformers library code
First, locate directory of your transformer library

    import transformers
    TRANSFORMERS_PATH = transformers.__path__
Replace these two files two files with corresponding files in transformers_files directory of this repo

    TRANSFORMERS_PATH/models/llama/modeling_llama.py
    TRANSFORMERS_PATH/generation/utils.py

Make back up as needed. The modified files keep everything in original files unchanged (so after replacement anything that works with original file will work in replaced files) and add additional functions for surgery. 

# File changes made

`utils.py`
- Added `generate_interpret`. A modified version of `generate` in the same file.
-  Added `greedy_search_interpret`. A modified version of `greedy_search` in the same file.
   -  Note: generation with surgery is currently only implemented for greedy search.

`modeling_llama.py`
- Added `forward_interpret`. A modified version of `forward` in the same file.

# Documentations

Suppose have you loaded llama models with

    tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")


## Single forward pass with surgery

Calling `model.forward_interpret` does same thing as calling `model.forward` with additional capability of doing surgery. 

`model.forward_interpret` takes in same arguments as `model.forward` with an additional parameter `insert_info`. 

If `insert_info` is `None`, `model.forward_interpret` does same thing as `model.forward`. If `insert_info` is not `None`, `model.forward_interpret` will do surgery on the forward pass.

## Generate sequence with surgery

Calling `model.generate_interpret` does same thing as calling `model.generate` with additional capability of doing surgery. 

It takes in same additional argument `insert_info`. If `insert_info` is `None`, `model.generate_interpret` does same thing as `model.generate`. If `insert_info` is not `None`, `model.generate_interpret` will do surgery on the forward pass.

## Format for `insert_info`

`insert_info` is a list of dictionary. i-th element in the list provides surgery specification for i-th sample in the batch. Each dictionary has following keys

- Any integer from 0 to number of layers-1 (tuple): 
    - The tuple contains location of surgery and embedding used for surgery for the layer given by the key `(insert_locations, embedding_to_insert)`
    - `insert_locations` (list): a list of integer indices to insert the embedding.
    - `embedding_to_insert` (torch.FloatTensor): an embedding to insert. It has the shape [1, `len(insert_locations)`, `embedding_size`]. `embedding_to_insert[0, j, :]` is the embedding to insert at `insert_locations[j]` on this layer.
- `overlay_strength` (float): 
  - a float in the range [0,1] if `replacing_mode` is `"normalized"`
  - a positive float if `replacing_mode` is `"addition"`. This is the strength of the surgery.
- `replacing_mode` (str): `"nomralized"` or `"adddition"`.
  - `"normalized"`: 
    
        new_embedding = overlay_strength * embedding_to_insert + (1-overlay_strength) * old_embedding

  - `"addition"`:

        new_embedding = overlay_strength * embedding_to_insert + old_embedding


# Examples

- `scripts/interpretation_demo.ipynb` walks through using the surgery to interpret the model.
- `scripts/get_interpretations_from_examples.py`: get interpretations with examples in a specified json file

    - Usage: #TODO: wrap everything in arguments

            python get_interpretations_from_examples.py 

    - Example json file format: 

            {
                "examples":[
                    {
                        "prompt": "What is highest mountain in the world?"
                        "category": "fact",
                        "raw": true
                    },
                    ...
                ]
            }
    
        - `prompt` (str): prompt to generate from
        - `category` (str): category name of the prompt. Included in output file to identify the prompt.
        - `raw` (bool): If `raw` is not provided or is false, prompt "PROMPT" will be changed to "[INST] PROMPT [/INST]". Otherwise, prompt will be used as is.

    - Output csv file format: the output csv file includes many columns, the most important ones are
      - `name`: automatically formatted as `category_idx` for identifying prompt
      - `prompt`: the prompt interpreted
      - `prompt_output`: the original output generated from passing prompt into model
      - `layer_idx`: the index of layer on which the token being interpreted
      - `token_idx`": the index of the token being interpreted
      - `repeat_prompt`: interpretation prompt used
      - `repeat_prompt_name`: name of interpretation prompt 
      - `interpretation`: a list of `token_ids` of interpretation
      - `interpretation_weights`: relevancy score of each interpretation output token 
      - `interpretation_prob`: logits of each interpretation token 




