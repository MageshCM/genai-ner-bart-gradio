## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:

  In modern natural language processing (NLP), the identification of key entities such as names, organizations, locations, and temporal expressions from unstructured text is essential for downstream applications like information retrieval, document classification, and knowledge graph construction.
However, traditional rule-based or shallow machine learning approaches often lack the contextual understanding required for accurate entity recognition in complex sentences.
This project aims to build a prototype NER system using a fine-tuned BART (Bidirectional and Auto-Regressive Transformer) model, which effectively captures both contextual dependencies and semantic relationships. The model output is integrated with an interactive Gradio interface that allows real-time user testing, visualization, and performance assessment.

### DESIGN STEPS:

#### STEP 1:

Choose a fine-tuned BART model for Named Entity Recognition and set up the development environment with required libraries like transformers, torch, and gradio.

#### STEP 2:

Connect the model via the Hugging Face API or local pipeline, implement a function to process text inputs and retrieve entity predictions, and refine the token outputs for accurate labeling.

#### STEP 3:

Design an interactive Gradio interface with text input and highlighted output for entity visualization, add example prompts for testing, and deploy the prototype for real-time user evaluation.

### PROGRAM:
```
import os
import json
import requests
import gradio as gr
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
hf_api_key = os.environ['HF_API_KEY']
API_URL = os.environ['HF_API_NER_BASE']

def get_completion(inputs, parameters=None, ENDPOINT_URL=API_URL):
    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }
    data = {"inputs": inputs}
    if parameters:
        data["parameters"] = parameters
    response = requests.post(ENDPOINT_URL, headers=headers, json=data)
    if response.status_code != 200:
        raise ValueError(f"Model API returned {response.status_code} error: {response.text}")
    text = response.text.strip()
    # Try single valid JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try line-separated JSON chunks
        for line in text.splitlines():
            try:
                return json.loads(line)
            except Exception:
                continue
        raise ValueError(f"Unable to parse model output: {text}")

def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        # Extract entity type without prefix
        ent_type = token['entity'].replace("B-", "").replace("I-", "")
        
        # Check if this token continues the previous entity
        if (merged_tokens and 
            token['entity'].startswith("I-") and 
            merged_tokens[-1]['entity_type'] == ent_type):
            # Merge with previous token
            last = merged_tokens[-1]
            word = token['word']
            # Handle subword tokens
            if word.startswith("##"):
                last['word'] += word[2:]
            else:
                # Add space before non-subword continuations
                last['word'] += " " + word
            last['end'] = token['end']
            last['score'] = (last['score'] + token['score']) / 2
        else:
            # Start new entity
            merged_tokens.append({
                "word": token['word'].replace("##", ""),
                "entity": token['entity'],
                "entity_type": ent_type,
                "start": token['start'],
                "end": token['end'],
                "score": token['score']
            })
    return merged_tokens

def ner(input_text):
    output = get_completion(input_text)
    if not isinstance(output, list):
        raise ValueError(f"Unexpected model output: {output}")
    
    merged_tokens = merge_tokens(output)
    results = []
    for ent in merged_tokens:
        results.append((ent['word'], ent['entity_type']))
    return results

gr.close_all()
demo = gr.Interface(
    fn=ner,
    inputs=gr.Textbox(label="Text to find entities", lines=2),
    outputs=gr.HighlightedText(label="Text with merged entities"),
    title="NER with dslim/bert-base-NER",
    description="Find named entities using the `dslim/bert-base-NER` model via Hugging Face Inference API.",
    allow_flagging="never",
    examples=[
        " I work at DeepLearningAI and live in Chennai.",
        
    ]
)
demo.launch(share=True, server_port=int(os.environ.get("PORT3", 7860)))
```

### OUTPUT:
<img width="1061" height="555" alt="image" src="https://github.com/user-attachments/assets/e6af01d7-046a-42b1-9a72-b2090d7be457" />


### RESULT:

The NER prototype using a fine-tuned BART model and Gradio was successfully developed to identify and highlight entities like names, places, and organizations.
It performs real-time text analysis with accurate and interactive visualization of recognized entities.
