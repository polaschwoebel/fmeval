import numpy as np
import pandas as pd
from llama_guard_prompts import TASK, INSTRUCTION, UNSAFE_CONTENT_CATEGORIES_OPENAI, UNSAFE_CONTENT_CATEGORIES_LLAMA, UNSAFE_CONTENT_CATEGORIES_GENDER
from typing import List
from sagemaker.jumpstart.model import JumpStartModel
import sagemaker

def retrieve_or_deploy_llama_guard(endpoint_name="meta-textgeneration-llama-guard-7b-2024-05-03-12-04-07-112"):
    # These are needed, even if you use an existing endpoint, by a cell later in this notebook.
    model_id, model_version = "meta-textgeneration-llama-guard-7b", "1.*"
    model = JumpStartModel(model_id=model_id, model_version=model_version)

    # check whether endpoint exists
    try:  
        predictor = sagemaker.predictor.Predictor(
            endpoint_name=endpoint_name,
            serializer=sagemaker.serializers.JSONSerializer(),
            deserializer = sagemaker.deserializers.JSONDeserializer()
        )
        # sample invocation to see whether 
        example_payload = model.retrieve_all_examples()[0]
        response = predictor.predict(example_payload.body)
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
    
# split data into train and test for few-shot learning
# from paper: 2 - 4 examples per category
# we will use 1 per category, and also  for no category (safe) to not run into input token limitation
def get_few_shot_examples(data):
    
    train_indices = []
    # sample the unsafe categories
    for category in ['S', 'H', 'V', 'HR', 'SH', 'S3', 'H2', 'V2']:
        category_train = data[data[category] == 1].sample(1, random_state=123).index
        train_indices.extend(category_train.values)
    
    # sample the safe category
    no_category_train = data[data[['S', 'H', 'V', 'HR', 'SH', 'S3', 'H2', 'V2']].sum(axis=1) == 0].sample(1, random_state=0).index
    train_indices.extend(no_category_train)

    test_indices = list(set(np.arange(data.shape[0])) - set(train_indices))

    train_data = data.iloc[train_indices]
    test_data = data.iloc[test_indices]
    return train_data, test_data


def format_few_shot_examples(train_data):
    examples = '\n\n<BEGIN EXAMPLES>'
    for _, row in train_data.iterrows():

        categories = ['S', 'H', 'V', 'HR', 'SH', 'S3', 'H2', 'V2']
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
    few_shot_examples: List[str]= []
) -> List[str]:
    """Format messages for Llama Guard models.

    Llama-guard models support 'user' and 'assistant' roles, and must alternate (u/a/u/a/u...). 
    For this experiment, we hardcode the roll to be user.
    """
    
    if len(few_shot_examples) == 0:
        examples = ""
    else:
        examples = format_few_shot_examples(few_shot_examples)

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