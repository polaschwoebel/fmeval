import numpy as np
import pandas as pd
from llama_guard_prompts import TASK, INSTRUCTION, UNSAFE_CONTENT_CATEGORIES_OPENAI, UNSAFE_CONTENT_CATEGORIES_LLAMA, UNSAFE_CONTENT_CATEGORIES_GENDER, UNSAFE_CONTENT_CATEGORY_NAMES_GENDER, UNSAFE_CONTENT_CATEGORY_NAMES_LLAMA, UNSAFE_CONTENT_CATEGORY_NAMES_OPENAI
from typing import List
from sagemaker.jumpstart.model import JumpStartModel
import sagemaker
from sagemaker.jumpstart.payload_utils import _construct_payload


def retrieve_or_deploy_llama(model_type="llama-guard"):
    # or  use "meta-textgeneration-llama-2-7b" for regular llama
    # These are needed, even if you use an existing endpoint, by a cell later in this notebook.
    model_id = "meta-textgeneration-llama-guard-7b" if model_type=='llama-guard' else "meta-textgeneration-llama-2-7b"
    endpoint_names = {"llama-guard": "meta-textgeneration-llama-guard-7b-2024-05-21-12-32-37-591", 
                     "llama": "meta-textgeneration-llama-2-7b-2024-05-10-11-29-11-532"}
    model_version = "1.2" if model_type=='llama-guard' else '4.1.0' # llama-guard 1.2 has problem with return_full_text
    model = JumpStartModel(model_id=model_id, model_version=model_version)

    # check whether endpoint exists
    try:  
        predictor = sagemaker.predictor.Predictor(
            endpoint_name=endpoint_names[model_type],
            serializer=sagemaker.serializers.JSONSerializer(),
            deserializer = sagemaker.deserializers.JSONDeserializer()
        )
        # sample invocation to see whether 
        # example_payload = model.retrieve_all_examples()[0]
        response = predictor.predict({"inputs": "Hello world! Today is"}) # example_payload.body)
        print('Found existing endpoint!')

    except:
        # To deploy via notebook
        accept_eula = True  # change to True to accept EULA for successful model deployment
        predictor = model.deploy(accept_eula=accept_eula)
        print('Deployed new endpoint!')

    return predictor


def delete_endpoint(predictor):
    try:
        predictor.delete_model()
        predictor.delete_endpoint()
    except Exception as e:
        print(e)
        

def get_unsafe_categories(taxonomy='llama-guard'):
    if taxonomy == 'llama-guard':
        unsafe_categories = UNSAFE_CONTENT_CATEGORIES_LLAMA
    elif taxonomy == 'openai-content-moderation':
        unsafe_categories = UNSAFE_CONTENT_CATEGORIES_OPENAI
    elif taxonomy == 'gender':
        unsafe_categories = UNSAFE_CONTENT_CATEGORIES_GENDER
    return unsafe_categories

def get_unsafe_category_names(taxonomy='llama-guard'):
    if taxonomy == 'llama-guard':
        unsafe_categories = UNSAFE_CONTENT_CATEGORY_NAMES_LLAMA
    elif taxonomy == 'openai-content-moderation':
        unsafe_categories = UNSAFE_CONTENT_CATEGORY_NAMES_OPENAI
    elif taxonomy == 'gender':
        unsafe_categories = UNSAFE_CONTENT_CATEGORY_NAMES_GENDER
    return unsafe_categories

 
# split data into train and test for few-shot learning
# from paper: 2 - 4 examples per category
# we will use 1 per category, and also  for no category (safe) to not run into input token limitation
def get_few_shot_examples(data, dataset_name):
    categories = get_unsafe_category_names(taxonomy=dataset_name)

    train_indices = []
    # sample the unsafe categories
    for category in categories:
        category_train = data[data[category] == 1].sample(2, random_state=123).index
        train_indices.extend(category_train.values)
    
    # sample the safe category
    no_category_train = data[data[categories].sum(axis=1) == 0].sample(2, random_state=0).index
    train_indices.extend(no_category_train)

    test_indices = list(set(np.arange(data.shape[0])) - set(train_indices))

    train_data = data.iloc[train_indices]
    test_data = data.iloc[test_indices]
    return train_data, test_data


def format_few_shot_examples(train_data, dataset_name):
    examples = '\n\n<BEGIN EXAMPLES>'
    for _, row in train_data.iterrows():

        categories = get_unsafe_category_names(taxonomy=dataset_name)
        labels = [category for category in categories if row[category]==1]
        if len(labels) == 0:
            binary_label = 'safe'
            examples += f"\nUser:\n{row.prompt}\nLabel: \n{binary_label}\n"
        else:
            binary_label = 'unsafe'
            examples += f"\nUser:\n{row.prompt}\nLabel: \n{binary_label}\n{labels}\n \n"
    examples +=  '<END EXAMPLES>\n'
    return examples   

def format_guard_messages(
    prompt: str,
    task: str = TASK,
    instruction: str = INSTRUCTION,
    unsafe_content_categories: str = UNSAFE_CONTENT_CATEGORIES_LLAMA,
    few_shot_examples: List[str]= [],
    dataset_name: str = 'openai-content-moderation'
) -> List[str]:
    """Format messages for Llama Guard models.

    Llama-guard models support 'user' and 'assistant' roles, and must alternate (u/a/u/a/u...). 
    For this experiment, we hardcode the roll to be user.
    """
    
    if len(few_shot_examples) == 0:
        examples = ""
    else:
        examples = format_few_shot_examples(few_shot_examples, dataset_name)

    message =  "".join(
        [
            "[INST] ",
            task,
            "\n\n<BEGIN UNSAFE CONTENT CATEGORIES>\n",
            unsafe_content_categories,
            "\n<END UNSAFE CONTENT CATEGORIES>",
            examples,
            "\n\n<BEGIN CONVERSATION>\n\n"
            f"User: {prompt}",
            "\n\n<END CONVERSATION>\n\n",
            instruction,
            "[/INST]"
        ]
    )
        
    return message


def parse_output_strict(response, dataset_category_names):
    if response.strip().lower().startswith('safe'):
        response_binary = 'safe'
        response_category = ['']
    elif response.strip().lower().startswith('unsafe'):
        response_binary = 'unsafe'
        response_category = []
        for category in dataset_category_names:
            if category in response:
                response_category.append(category)
    else:
        print('invalid answer')
        response_binary = 'invalid'
        response_category = ['']
    return response_binary, response_category


# def parse_output_fuzzy(response, dataset_category_names):
#     if 'unsafe' in response.lower():
#         response_binary = 'unsafe'
#         response_category = []
#         for category in dataset_category_names:
#             if category in response:
#                 response_category.append(category)
                
#     elif 'safe' in response.lower():
#         response_binary = 'safe'
#         response_category = ['']
        
#     else:
#         print('invalid answer')
#         print(response)
#         response_binary = 'invalid'
#         response_category = ['']
#     return response_binary, response_category