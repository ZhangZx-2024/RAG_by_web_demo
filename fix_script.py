from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from googlesearch import search
import requests
from bs4 import BeautifulSoup
import torch
import warnings
warnings.filterwarnings("ignore")

# Load the LLaMA model (this part works based on your output)
print("Loading model and tokenizer...")
model_name = "mediocredev/open-llama-3b-v2-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./cache')
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='./cache', torch_dtype=torch.bfloat16).to('cuda')

pipeline = transformers.pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map='auto',
)

# Use all the same functions from your original script
def generate_search_query(user_input):
    System_prompt = "Convert the user's input into a search query suitable for web search."
    prompt = f'### System:\n{System_prompt}<|endoftext|>\n### User:\nConvert my following content into a search query suitable for web search:\n{user_input}<|endoftext|>\n### Assistant:\n'
    print("Prompt length: ", len(prompt))
    response = pipeline(
        prompt,
        max_new_tokens=100,
        repetition_penalty=1.05,
    )
    response = response[0]['generated_text']
    assistant_index = response.find("Assistant:") + len("Assistant:")
    result = response[assistant_index:].strip()
    return result

def search_web(query):
    search_results = []
    query += " after:2025-01-01"
    for url in search(query, num_results=10):
        if url:
            search_results.append(url)
        if len(search_results) == 3:
            break
    return search_results

def extract_main_content(url):
    response = requests.get(url, verify=False)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    main_content = ' '.join([para.get_text() for para in paragraphs])
    return main_content

def generate_response(user_input, main_contents):
    System_prompt = f"Based on the following web content, provide a response to the user's question:\n\n"
    main_contents = [content[:3000] for content in main_contents]
    System_prompt += "\n\n".join(main_contents)
    
    prompt = f'### System:\n{System_prompt}<|endoftext|>\n### User:\n{user_input}<|endoftext|>\n### Assistant:\n'
    print("Prompt length: ", len(prompt))
    response = pipeline(
        prompt,
        max_new_tokens=1000,
        repetition_penalty=1.05,
    )
    response = response[0]['generated_text']
    assistant_index = response.find("Assistant:") + len("Assistant:")
    result = response[assistant_index:].strip()
    return result

def generate_response_no_RAG(user_input):
    System_prompt = "Provide a response to the user's question."
    prompt = f'### System:\n{System_prompt}<|endoftext|>\n### User:\n{user_input}<|endoftext|>\n### Assistant:\n'
    print("Prompt length: ", len(prompt))
    response = pipeline(
        prompt,
        max_new_tokens=1000,
        repetition_penalty=1.05,
    )
    response = response[0]['generated_text']
    assistant_index = response.find("Assistant:") + len("Assistant:")
    result = response[assistant_index:].strip()
    return result

# Instead of user input, use predefined questions
print("Running RAG demo with predefined questions...")
questions = [
    "What is deep learning?",
    "How do transformers work in NLP?",
    "Explain the concept of attention mechanism"
]

# Process each question
for question in questions:
    print("\n\n" + "="*60)
    print(f"QUESTION: {question}")
    print("="*60 + "\n")
    
    print("Generating query...")
    search_query = generate_search_query(question)
    
    print("Generating response with no RAG...")
    answer = generate_response_no_RAG(question)
    
    print("Searching...")
    search_results = search_web(search_query)
    
    print("Extracting main content...")
    main_contents = []
    for url in search_results:
        main_content = extract_main_content(url)
        main_contents.append(main_content)
    
    print("Generating final response with RAG...")
    response = generate_response(question, main_contents)
    
    print("\n" + "="*60)
    print(f"RESULTS FOR: {question}")
    print(f"Search query: {search_query}\n")
    print(f"Model's response with no RAG:\n{answer}\n\n")
    print(f"Final response with RAG:\n{response}\n\n")
    print(f"Reference: {search_results}")
    print("="*60)

print("\nRAG demo completed successfully!")