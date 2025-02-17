import os
import requests
import torch
from bs4 import BeautifulSoup
import json
import schedule
import time
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

# Constants for model directory
MODEL_CACHE_DIR = '.models'
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)  # Create the model directory if it doesn't exist

def process_html_content_with_llm(html_content, max_new_tokens):
    """
    Process HTML content using a local language model with caching.
    
    Parameters:
        html_content (str): The HTML content to be processed.
        max_new_tokens (int): The maximum number of new tokens to generate.
    
    Returns:
        str: The processed content as a text string.
    """
    # Check device availability and load model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = "jinaai/reader-lm-1.5b"

    # Load model and tokenizer with specified cache directory
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=MODEL_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, cache_dir=MODEL_CACHE_DIR).to(device)

    # Prepare the input text for the model
    messages = [{"role": "user", "content": html_content}]
    input_text = tokenizer.convert_chat_template(messages, tokenize=False)

    # Encode input and generate the output sequence
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=max_new_tokens, temperature=0, do_sample=False, repetition_penalty=1.08)

    # Decode and return the generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_hacker_news_story(story, max_tokens):
    # Initialize content variables
    story_content = ''
    comments_content = ''
    
    try:
        # Fetch the story webpage
        story_response = requests.get(story['hacker_news_url'])
        if story_response.ok:
            soup = BeautifulSoup(story_response.text, 'html.parser')
            html_content = soup.prettify()
            # Process main story content
            story_content = process_html_content_with_llm(html_content, max_new_tokens=max_tokens)
        else:
            print(f"Failed to fetch story: {story_response.status_code}")

        # Fetch comments webpage
        comments_response = requests.get(f"https://news.ycombinator.com/item?id={story['id']}")
        if comments_response.ok:
            comments_soup = BeautifulSoup(comments_response.text, 'html.parser')
            comments = comments_soup.select('.comment')[:10]  # Limit to first 10 comments for example
            comments_html_content = '\n'.join(comment.prettify() for comment in comments)
            # Process comments content
            comments_content = process_html_content_with_llm(comments_html_content, max_new_tokens=max_tokens)
        else:
            print(f"Failed to fetch comments: {comments_response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

    # Assemble full content from story and comments
    content_parts = []
    if story['title']:
        content_parts.append(f"<title>\n{story['title']}\n</title>")
    if story_content:
        content_parts.append(f"<article>\n{story_content}\n</article>")
    if comments_content:
        content_parts.append(f"<comments>\n{comments_content}\n</comments>")

    return '\n\n---\n\n'.join(content_parts)

# Fetch the Hacker News stories
def fetch_hacker_news():
    url = "https://news.ycombinator.com"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    stories = []
    for item in soup.select('.athing'):
        title = item.select_one('.titleline > a').text
        story_url = item.select_one('.titleline > a')['href']
        story_id = item.get('id')
        hacker_news_url = f"https://news.ycombinator.com/item?id={story_id}"

        next_row = item.find_next_sibling('tr')
        points = next_row.select_one('.score')
        points = points.text if points else '0 points'
        comments = next_row.select_one('a[href^="item?id="]')
        comments = comments.text if comments else '0 comments'

        stories.append({
            'id': story_id,
            'title': title,
            'url': story_url,
            'hacker_news_url': hacker_news_url,
            'points': points,
            'comments': comments
        })

    return stories

# Save the JSON data to a file
def save_to_json(data):
    timestamp = time.strftime("%Y%m%d_%H%M")
    date = time.strftime("%Y%m%d")
    if not os.path.exists(date):
        os.makedirs(date)
    filename = f'{date}/hacker_news_stories_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# Define the job to fetch stories and add detailed content
def job(max_tokens):
    print("Fetching Hacker News stories...")
    stories = fetch_hacker_news()
    
    # Fetch full story content using Jina and add to story data
    for story in stories:
        story_content = get_hacker_news_story(story, max_tokens)
        story['content'] = story_content
    
    save_to_json(stories)

    for story in stories:
        print(f"Title: {story['title']}")
        print(f"URL: {story['url']}")
        print(f"Hacker News URL: {story['hacker_news_url']}")
        print(f"Points: {story['points']}")
        print(f"Comments: {story['comments']}")
        print(f"Content: {story['content'][:200]}...")  # Print first 200 characters for brevity
        print("-" * 50)

# Schedule the job to run at specific intervals
def schedule_job(interval, max_tokens):
    print(f"Scheduling job to run every {interval} minute(s)...")
    schedule.every(interval).minutes.do(job, max_tokens=max_tokens)

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Hacker News stories at a specified interval.")
    parser.add_argument('--interval', type=int, default=1, help='Interval in minutes to fetch stories')
    parser.add_argument('--max-tokens', type=int, default=1000, help='Maximum token length for story content')
    args = parser.parse_args()
    
    schedule_job(args.interval, args.max_tokens)
