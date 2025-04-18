from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from googlesearch import search
import requests
from bs4 import BeautifulSoup
import torch
import warnings
import re
import os
import time
from tqdm import tqdm
warnings.filterwarnings("ignore")

# Load a model with longer context window
print("Loading model and tokenizer...")
model_name = "mediocredev/open-llama-3b-v2-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./cache')
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='./cache', torch_dtype=torch.bfloat16).to('cuda')

# Set token limits for context
MAX_TOKENS = 1000  # Reduced from 1500 to be well under model's 2048 token limit
TOKEN_MARGIN = 300  # Increased safety margin

pipeline = transformers.pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map='auto',
)

def generate_search_query(user_input):
    System_prompt = "Convert the user's input into a search query suitable for web search."
    prompt = f'{System_prompt} Convert my following content into a search query suitable for web search: {user_input}'
    
    tokens = len(tokenizer.encode(prompt))
    print(f"Prompt length: {tokens} tokens")
    
    if tokens > MAX_TOKENS:
        print(f"Warning: Input too long ({tokens} tokens), truncating...")
        # Truncate the input to fit within token limits
        truncated_input = tokenizer.decode(tokenizer.encode(user_input)[:MAX_TOKENS-100])
        prompt = f'{System_prompt} Convert my following content into a search query suitable for web search: {truncated_input}'
    
    try:
        response = pipeline(
            prompt,
            max_new_tokens=100,
            repetition_penalty=1.05,
        )
        response = response[0]['generated_text']
        result = response.split('</s>')[1].strip()
        return result
    except Exception as e:
        print(f"Error generating search query: {e}")
        return user_input  # Fallback to using the original input as search query

def search_web(query):
    search_results = []
    query += " after:2025-01-01"
    try:
        # Add timeout for search to prevent hanging
        search_generator = search(query, num_results=10, timeout=10)
        
        # Use a timeout to prevent hanging
        import signal
        class TimeoutException(Exception): pass
        
        def timeout_handler(signum, frame):
            raise TimeoutException("Search timed out")
        
        # Set the timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(15)  # 15 second timeout
        
        try:
            for url in search_generator:
                if url:
                    search_results.append(url)
                if len(search_results) == 3:
                    break
            
            # Cancel the timeout
            signal.alarm(0)
        except TimeoutException:
            print("Search operation timed out, proceeding with results found so far")
        except Exception as e:
            print(f"Error during search iteration: {e}")
        
    except Exception as e:
        print(f"Error in web search: {e}")
    
    # If no results, return empty list but don't hang
    print(f"Found {len(search_results)} search results")
    return search_results

def truncate_content(content, max_tokens):
    """Truncate content to fit within token limit"""
    tokens = tokenizer.encode(content)
    if len(tokens) <= max_tokens:
        return content
    
    # Truncate and add indication that content was cut
    truncated_tokens = tokens[:max_tokens-10]  # Leave room for ellipsis
    truncated_content = tokenizer.decode(truncated_tokens) + "..."
    print(f"Truncated content from {len(tokens)} to {len(truncated_tokens)} tokens")
    return truncated_content

def extract_main_content(url):
    try:
        # Add timeout to requests to prevent hanging
        response = requests.get(url, verify=False, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        main_content = ' '.join([para.get_text() for para in paragraphs])
        
        # Limit content size to avoid token overflow - more aggressive truncation
        content_tokens = len(tokenizer.encode(main_content))
        max_allowed = MAX_TOKENS // 4  # Even smaller chunk per source to be safe
        
        if content_tokens > max_allowed:
            print(f"Content too large ({content_tokens} tokens), truncating to {max_allowed}...")
            return truncate_content(main_content, max_allowed)
        
        return main_content
    except requests.exceptions.Timeout:
        print(f"Request to {url} timed out")
        return ""
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return ""

def generate_response(user_input, main_contents):
    # More aggressive content limitation
    if len(main_contents) > 2:
        print("Too many content sources, limiting to top 2")
        main_contents = main_contents[:2]
    
    # Estimate token usage and limit content if needed
    total_content = "\n\n".join(main_contents)
    content_tokens = len(tokenizer.encode(total_content))
    
    max_content_tokens = MAX_TOKENS - TOKEN_MARGIN
    if content_tokens > max_content_tokens:
        print(f"Warning: Content too large ({content_tokens} tokens), truncating to {max_content_tokens}...")
        # Simple truncation strategy - reduce each content proportionally
        ratio = max_content_tokens / content_tokens
        
        truncated_contents = []
        for content in main_contents:
            content_tokens = len(tokenizer.encode(content))
            target_tokens = max(100, int(content_tokens * ratio))
            truncated_contents.append(truncate_content(content, target_tokens))
        
        main_contents = truncated_contents
        total_content = "\n\n".join(main_contents)
    
    # Final check to ensure we're under the limit
    if len(tokenizer.encode(total_content)) > max_content_tokens:
        print("Content still too large after truncation, using more aggressive approach")
        total_content = truncate_content(total_content, max_content_tokens)
    
    # Keep prompt simple and short
    System_prompt = f"Based on this web content, answer the user's question:"
    
    # Check final prompt size
    prompt = f'{System_prompt}\n\n{total_content}\n\nQuestion: {user_input}'
    tokens = len(tokenizer.encode(prompt))
    print(f"Final prompt length: {tokens} tokens")
    
    # Last-resort truncation if still too long
    if tokens > MAX_TOKENS:
        available_tokens = MAX_TOKENS - len(tokenizer.encode(f'{System_prompt}\n\nQuestion: {user_input}')) - 100
        if available_tokens > 200:  # Only proceed if we have enough space for useful content
            print(f"Final truncation to {available_tokens} tokens for content")
            total_content = truncate_content(total_content, available_tokens)
            prompt = f'{System_prompt}\n\n{total_content}\n\nQuestion: {user_input}'
        else:
            # Fall back to no RAG if we can't fit content
            print("Not enough space for RAG content, falling back to direct question answering")
            return generate_response_no_RAG(user_input)
    
    try:
        response = pipeline(
            prompt,
            max_new_tokens=1000,
            repetition_penalty=1.05,
        )
        response = response[0]['generated_text']
        result = response.split('</s>')[1].strip()
        return result
    except Exception as e:
        print(f"Error generating response with RAG: {e}")
        return "Error generating response with RAG."

def generate_response_no_RAG(user_input):
    System_prompt = "Answer this question concisely:"
    prompt = f'{System_prompt} {user_input}'
    
    tokens = len(tokenizer.encode(prompt))
    print(f"Prompt length: {tokens} tokens")
    
    if tokens > MAX_TOKENS:
        print(f"Warning: Input too long ({tokens} tokens), truncating...")
        # Truncate the input to fit within token limits
        truncated_input = tokenizer.decode(tokenizer.encode(user_input)[:MAX_TOKENS-100])
        prompt = f'{System_prompt} {truncated_input}'
    
    try:
        response = pipeline(
            prompt,
            max_new_tokens=1000,
            repetition_penalty=1.05,
        )
        response = response[0]['generated_text']
        result = response.split('</s>')[1].strip()
        return result
    except Exception as e:
        print(f"Error generating response without RAG: {e}")
        return "Error generating response without RAG."

def read_test_dataset(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Extract questions using regex
    questions = []
    matches = re.findall(r'\*\*Question \d+: (.*?)\*\*\n Correct Answer: (.*?)\n Potential Incorrect Answer: (.*?)\n', content, re.DOTALL)
    
    for match in matches:
        question_text = match[0].strip()
        correct_answer = match[1].strip()
        incorrect_answer = match[2].strip()
        questions.append({
            "question": question_text,
            "correct_answer": correct_answer,
            "incorrect_answer": incorrect_answer
        })
    
    return questions

def process_question(question_data, output_dir):
    """Process a single question and save results to a file"""
    user_input = question_data["question"]
    
    # Fix the ID extraction logic that was causing the error
    if "id" in question_data:
        question_id = question_data["id"].replace("Question ", "")
    else:
        question_id = "unknown"
    
    output_file = os.path.join(output_dir, f"question_{question_id}_results.txt")
    
    # Check if this question was already processed
    if os.path.exists(output_file):
        print(f"Question {question_id} already processed, skipping...")
        return
    
    print("\n\n\n")
    print(f"Question: {user_input}")
    print(f"Correct Answer: {question_data['correct_answer']}")
    
    results = []
    results.append(f"Question: {user_input}")
    results.append(f"Correct Answer: {question_data['correct_answer']}")
    
    # Generate search query with timeout protection
    print("Generating query...")
    try:
        import signal
        class TimeoutException(Exception): pass
        
        def timeout_handler(signum, frame):
            raise TimeoutException("Operation timed out")
        
        # Set the timeout for query generation
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30 second timeout
        
        search_query = generate_search_query(user_input)
        
        # Cancel the timeout
        signal.alarm(0)
    except TimeoutException:
        print("Query generation timed out, using original question")
        search_query = user_input
    except Exception as e:
        print(f"Error in query generation: {e}, using original question")
        search_query = user_input
    
    results.append(f"Search query: {search_query}")
    
    # Generate response without RAG with timeout protection
    print("Generating response with no RAG...")
    try:
        signal.alarm(60)  # 60 second timeout for model inference
        answer = generate_response_no_RAG(user_input)
        signal.alarm(0)
    except TimeoutException:
        print("Model inference timed out for no-RAG response")
        answer = "Response generation timed out."
    except Exception as e:
        print(f"Error in no-RAG generation: {e}")
        answer = f"Error generating response: {str(e)}"
    
    results.append(f"Model's response with no RAG:\n{answer}")
    
    # Search the web with built-in timeout
    print("Searching...")
    search_results = search_web(search_query)
    results.append(f"Search results: {', '.join(search_results)}")
    
    # Extract content with built-in timeout
    print("Extracting main content...")
    main_contents = []
    for url in search_results:
        main_content = extract_main_content(url)
        if main_content:  # Only add if content was extracted
            main_contents.append(main_content)
    
    # Generate RAG response if we have content
    if main_contents:
        print("Generating final response with RAG...")
        try:
            signal.alarm(60)  # 60 second timeout for RAG inference
            response = generate_response(user_input, main_contents)
            signal.alarm(0)
        except TimeoutException:
            print("Model inference timed out for RAG response")
            response = "RAG response generation timed out."
        except Exception as e:
            print(f"Error in RAG generation: {e}")
            response = f"Error generating RAG response: {str(e)}"
        
        results.append(f"Final response with RAG:\n{response}")
    else:
        results.append("No web content found for RAG")
    
    # Save results to file
    with open(output_file, 'w') as f:
        f.write("\n\n".join(results))
    
    # Print results
    print("########################################")
    print(f"User input: {user_input}")
    print(f"Correct Answer: {question_data['correct_answer']}")
    print(f"Search query: {search_query}\n")
    print(f"Model's response with no RAG:\n{answer}\n\n")
    
    if main_contents:
        print(f"Final response with RAG:\n{response}\n\n")
    else:
        print("No web content found for RAG\n\n")
        
    print(f"Reference: {search_results}")
    print("########################################")

# Create output directory if it doesn't exist
output_dir = "question_results"
os.makedirs(output_dir, exist_ok=True)

print("RAG demo ready. Processing test dataset in batches...")

# Read questions from the test dataset
test_questions = read_test_dataset("Test dataset.md")
print(f"Loaded {len(test_questions)} questions from dataset")

# Add question IDs for tracking
for i, question in enumerate(test_questions):
    question["id"] = f"Question {i+1}"

# Process questions in batches to avoid memory issues
BATCH_SIZE = 5  # Process 5 questions at a time
for i in range(0, len(test_questions), BATCH_SIZE):
    batch = test_questions[i:i+BATCH_SIZE]
    print(f"\nProcessing batch {i//BATCH_SIZE + 1} of {(len(test_questions)-1)//BATCH_SIZE + 1}")
    
    for question_data in tqdm(batch):
        try:
            # Set a global timeout for the entire question processing
            import signal
            
            def global_timeout_handler(signum, frame):
                print(f"Processing question {question_data.get('id', 'unknown')} timed out, moving to next")
                raise TimeoutException("Question processing timed out")
            
            signal.signal(signal.SIGALRM, global_timeout_handler)
            signal.alarm(180)  # 3 minute timeout per question
            
            process_question(question_data, output_dir)
            
            # Cancel the timeout
            signal.alarm(0)
            
            # Small delay to avoid rate limiting
            time.sleep(2)
        except Exception as e:
            print(f"Error processing question {question_data.get('id', 'unknown')}: {e}")
            # Continue with next question

print("\nRAG demo completed successfully!")
print(f"Results saved to {output_dir}/")