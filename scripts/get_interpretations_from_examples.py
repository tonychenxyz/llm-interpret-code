import os
import numpy as np
import torch
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,5,7"
print(torch.cuda.device_count())
from transformers import AutoTokenizer,AutoModelForCausalLM
# model_path = "/local/vondrick/hc3295/converted_weights_llama_chat7b"#
# model_path = "/proj/vondrick3/bigmodels/llama2_chat/converted_weights_llama_chat7b"
model_path = "/proj/vondrick3/bigmodels/llama2_chat/converted_weights_llama_chat70b"
# model_path = "/local/vondrick/hc3295/converted_weights_llama_chat70b"

tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

bs = 64
out_csv_name = '1120guess.csv'
repeat_prompts = [("[INST] _ _ _ _ _ [/INST]\nSure, I'll summerize your message:", 4, 9, 'summerize'),
                # ("[INST] _ [/INST] I think you said:", 4, 5, 'I think you said'),
                ("[INST] _ _ _ _ _ [/INST] Sure, I will repeat for you:", 4, 9, 'repeat'),
                ("[INST] _ _ _ _ _ [/INST] The object you said is:", 4, 9, 'object'),
                ("[INST] _ _ _ _ _ [/INST] The object in your message is:", 4, 9, 'object_you_said')]
prompt_output_length = 1 #50

import torch
import json
import pandas as pd
import os

with open('examples_config.json', 'r') as f:
    config = json.load(f)
examples = config['examples']

examples.sort(key=lambda x: len(x['prompt']))


category_counts = {}

if False:#out_csv_name in os.listdir():
    df = pd.read_csv(out_csv_name)
    result_df = df.to_dict('list')
else:
    result_df = {
                'name': [],
                'prompt': [],
                'current token': [],
                'token_idx': [],
                'interpretation': [],
                'prompt_output': [],
                'prompt_token_list': [],
                'category': [],
                'layer_idx': [],   
                'interpretation_weights': [],
                'interpretation_prob': [],
                'repeat_prompt': [],
                'repeat_prompt_name': [],
            }


with torch.no_grad():
    for example in examples:
        if example['category'] not in category_counts:
                category_counts[example['category']] = 1
                prompt_name = f"{example['category']}_1"
        else:
            category_counts[example['category']] += 1
            prompt_name = f"{example['category']}_{category_counts[example['category']]}"
        try:
                raw = example['raw']
                prompt= example['prompt']
        except KeyError:
            prompt = f"[INST] {example['prompt']} [/INST]"
                
        print('getting prompt output')
        model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        output = model.generate(**model_inputs, max_new_tokens=prompt_output_length)
        prompt_out = tokenizer.decode(output[0], skip_special_tokens=True)
        cropped_prompt_out = prompt_out[len(prompt)+1:]
        model_inputs = tokenizer(prompt_out, return_tensors="pt").to("cuda:0")
        print('getting embeddings')
        outputs = model.forward_interpret(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=True,
                    output_hidden_states=True,
                )
        
        prompt_len = model_inputs['input_ids'].shape[-1]
        prompt_tokenized = tokenizer.convert_ids_to_tokens(model_inputs['input_ids'][0])
        prompt_tokenids = model_inputs['input_ids'][0]
        
        for repeat_prompt, start_insert, end_insert, repeat_prompt_name in repeat_prompts:            
            
            
            all_insert_infos = []
            for retrieve_layer in [0, 1, 5, 10,15,20,25,30,35,40,45,50,55,60,65,70,75,79, 80]:
                for retrieve_token in range(prompt_len):
                    insert_info = {}
                    insert_info['replacing_mode'] = 'normalized'
                    insert_info['overlay_strength'] = 1
                    insert_info['retrieve_layer'] = retrieve_layer
                    insert_info['retrieve_token'] = retrieve_token
                    for idx, layer in enumerate(model.model.layers):
                        if idx == 1:
                            insert_locations = [i for i in range(start_insert, end_insert)]
                            insert_info[idx] = (insert_locations, outputs['hidden_states'][retrieve_layer][0][retrieve_token].repeat(1,len(insert_locations), 1))
                    all_insert_infos.append(insert_info)


            for batch_start_idx in range(0,len(all_insert_infos),bs):
                with torch.no_grad():
                    print(batch_start_idx)
                    batch_insert_infos = all_insert_infos[batch_start_idx:min(batch_start_idx+bs, len(all_insert_infos))]

                    model_inputs = tokenizer([repeat_prompt for i in range(len(batch_insert_infos))], return_tensors="pt").to("cuda:0")
                    repeat_prompt_n_tokens = model_inputs['input_ids'].shape[-1]

                    output = model.generate_interpret(**model_inputs, max_new_tokens=30, insert_info=batch_insert_infos, pad_token_id=tokenizer.eos_token_id, output_attentions = False)


                    original_attribution_outputs = model.forward_interpret(
                                            output[...,:-1],
                                            return_dict=True,
                                            output_attentions=True,
                                            output_hidden_states=True,
                                            insert_info=batch_insert_infos,
                                        )

                    original_final_logits = model.lm_head(original_attribution_outputs.hidden_states[-1])
                    original_final_logits = torch.softmax(original_final_logits, dim=-1)

                    corrupted_attribution_outputs = model.forward_interpret(
                                    output[...,:-1],
                                    return_dict=True,
                                    output_attentions=True,
                                    output_hidden_states=True,
                                )

                    corrupted_final_logits = model.lm_head(corrupted_attribution_outputs.hidden_states[-1])
                    corrupted_final_logits = torch.softmax(corrupted_final_logits, dim=-1)



                    diff = (original_final_logits - corrupted_final_logits).abs()

                    indices = output[:,1:].detach().cpu().long()
                    indices = indices.unsqueeze(-1)
                    selected_diff = torch.gather(diff.detach().cpu(), 2, indices).squeeze(-1)

                    original_probs = torch.gather(original_final_logits.detach().cpu(), 2, indices).squeeze(-1)
                    cropped_original_probs = original_probs[:,repeat_prompt_n_tokens-1:]
                    interpretation_probs = cropped_original_probs#[:,:10].prod(dim=-1)

                    
                    cropped_selected_diff = selected_diff[:,repeat_prompt_n_tokens-1:]
                    cropped_interpretation_tokens = output[:,repeat_prompt_n_tokens:]

                    result_df['prompt'] += [prompt]*len(batch_insert_infos)
                    result_df['prompt_output'] += [cropped_prompt_out]*len(batch_insert_infos)
                    result_df['prompt_token_list'] += [prompt_tokenized]*len(batch_insert_infos)

                    result_df['name'] += [prompt_name]*len(batch_insert_infos)
                    result_df['category'] += [example['category']]*len(batch_insert_infos)
                    
                    result_df['layer_idx'] += [batch_insert_infos[i]['retrieve_layer'] for i in range(len(batch_insert_infos))]
                    result_df['token_idx'] += [batch_insert_infos[i]['retrieve_token'] for i in range(len(batch_insert_infos))]

                    result_df['interpretation'] += [list(cropped_interpretation_tokens[i].detach().cpu().numpy()) for i in range(len(batch_insert_infos))]
                    result_df['interpretation_weights'] += [list(cropped_selected_diff[i].detach().cpu().numpy()) for i in range(len(batch_insert_infos))]
                    result_df['current token'] += [prompt_tokenized[batch_insert_infos[i]['retrieve_token']] for i in range(len(batch_insert_infos))]

                    result_df['interpretation_prob'] += list(interpretation_probs.detach().cpu().numpy())

                    result_df['repeat_prompt'] += [repeat_prompt]*len(batch_insert_infos)
                    result_df['repeat_prompt_name'] += [repeat_prompt_name]*len(batch_insert_infos)
                    
                    pd.DataFrame(result_df).to_csv(out_csv_name, index=False)


            



            
    