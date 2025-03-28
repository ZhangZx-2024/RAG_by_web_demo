from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from googlesearch import search
import requests
from bs4 import BeautifulSoup
import torch
# hugging face logging in



# Load the LLaMA-v3.2-3B-Chat model from Hugging Face
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
    # 在搜索查询中加入时间过滤器，例如限制搜索结果为2025年之后的内容
    query += " after:2025-01-01"
    for url in search(query, num_results=10):
        if url:
            search_results.append(url)
        if len(search_results) == 3:
            break
    return search_results

def extract_main_content(url):
    response = requests.get(url,verify=False)
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



while(True):    
    user_input = input("Enter your question: ")
    if(user_input == "exit"):
        break
    print("\n\n\n")
    print("Generating query...")
    search_query = generate_search_query(user_input)
    print("Generating response with no RAG...")
    answer = generate_response_no_RAG(user_input)
    print("Searching...")
    search_results = search_web(search_query)
    print("Extracting main content...")
    main_contents = []
    for url in search_results:
        main_content = extract_main_content(url)
        main_contents.append(main_content)
    print("Generating final response with RAG...")
    response = generate_response(user_input, main_contents)
    print("########################################")
    print(f"User input: {user_input}")
    print(f"Search query: {search_query}\n")
    print(f"Model's response with no RAG:\n{answer}\n\n")
    
    print(f"Final response with RAG:\n{response}\n\n")
    print(f"Reference: {search_results}")
    print("########################################")

