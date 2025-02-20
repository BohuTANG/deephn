# export JINA_KEY=<your-jina-key>
# export OPENAI_BASE=<your-openai-base>
# export OPENAI_MODEL=DeepSeek-V3
# export OPENAI_API_KEY=<your-openai-api-key>
# export AZURE_SPEECH_KEY=<your-azure-speech-key>
# export AZURE_SPEECH_REGION=<your-region>

import os
import requests
from bs4 import BeautifulSoup
import json
import schedule
import time
import argparse
import openai
from typing import List
from urllib.parse import urljoin
import azure.cognitiveservices.speech as speechsdk

def check_env_var(env_var_name: str, error_message: str) -> str:
    value = os.getenv(env_var_name)
    if not value:
        raise ValueError(f"{error_message} is not set")
    return value

def summarize_content(content: str) -> str:
# Prompt templates
    SUMMARIZE_STORY_PROMPT = """
            You are an editorial assistant for the Hacker News podcast, skilled in transforming articles and comments from Hacker News into engaging podcast content. Your audience primarily consists of software developers and tech enthusiasts.
            【Objectives】  
            - Receive and read articles and comments from Hacker News.  
            - Provide a brief introduction to the main topic of the article, followed by a concise explanation of its key points.  
            - Analyze and summarize the diverse opinions in the comments section to showcase multiple perspectives.  
            - Communicate clearly and directly, as if having a simple and easy-to-understand conversation with a friend.
            - Avoid using any symbols such as **, *, #, etc.
            - Provide the content in both Chinese and English versions, with the English version first. Use --- to separate the content.
""".strip()


    try:
        client = openai.OpenAI(
            base_url=check_env_var('OPENAI_BASE', 'OpenAI base URL'),
            api_key=check_env_var('OPENAI_API_KEY', 'OpenAI API key')
        )
        
        response = client.chat.completions.create(
            model=check_env_var('OPENAI_MODEL', 'OpenAI model'),
            messages=[
                {"role": "system", "content": SUMMARIZE_STORY_PROMPT},
                {"role": "user", "content": content}
            ],
            max_tokens=1000,
            temperature=0.7,
            extra_headers={
                "HTTP-Referer": "https://news.ycombinator.com",
                "X-Title": "HN Podcast Assistant"
            }
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error summarizing content: {str(e)}")
        raise

def get_hacker_news_story(story, max_tokens):
    print(f"\nFetching content for story: {story['title']}")
    headers = {
        'X-Retain-Images': 'none',
    }

    jina_key = check_env_var('JINA_KEY', 'Jina key')
    story_content = ''
    comments_content = ''
    try:
        print(f"Fetching article from: {story['url']}")
        article_response = requests.get(f"https://r.jina.ai/{story['url']}", headers=headers)
        if article_response.ok:
            story_content = article_response.text[:max_tokens * 4]
            print("Successfully fetched article content")
        else:
            print(f"Get story failed: {article_response.status_code} {story['url']}")

        print(f"Fetching comments from: https://news.ycombinator.com/item?id={story['id']}")
        comments_response = requests.get(
            f"https://r.jina.ai/https://news.ycombinator.com/item?id={story['id']}", 
            headers={
                **headers,
                'X-Remove-Selector': '.navs',
                'X-Target-Selector': '#pagespace + tr'
            }
        )
        
        if comments_response.ok:
            comments_content = comments_response.text[:max_tokens * 4]
            print("Successfully fetched comments")
        else:
            print(f"Get story comments failed: {comments_response.status_code} https://news.ycombinator.com/item?id={story['id']}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

    content_parts = []
    if story['title']:
        content_parts.append(f"<title>\n{story['title']}\n</title>")
    if story_content:
        content_parts.append(f"<article>\n{story_content}\n</article>")
    if comments_content:
        content_parts.append(f"<comments>\n{comments_content}\n</comments>")

    return '\n\n---\n\n'.join(content_parts)

def fetch_hacker_news():
    url = "https://news.ycombinator.com"
    print(f"\nFetching stories from {url}")
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
        print(f"Found story: {title} ({points}, {comments})")

    print(f"Total stories found: {len(stories)}")
    return stories

def save_to_json(file_name: str, data):
    print(f"\nSaving stories to {file_name}")
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print("Stories saved successfully")

def text_to_speech(file_name: str, summary: str):
    # Initialize the speech configuration
    speech_key = check_env_var('AZURE_SPEECH_KEY', 'Azure Speech key')
    service_region = check_env_var('AZURE_SPEECH_REGION', 'Azure Speech region')
    print(f"Speech key: {speech_key}, Service region: {service_region}")
    
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    audio_config = speechsdk.audio.AudioOutputConfig(filename=file_name)

    # Synthesize speech from the summary text
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    print(f"Starting speech synthesis for file {file_name}...")
    result = speech_synthesizer.speak_text_async(summary).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"Speech synthesis completed successfully for {file_name}")
    else:
        print(f"Speech synthesis failed. Reason: {result.reason}")


def job(max_tokens, top_n):
    print("\n=== Starting Hacker News fetch job ===")
    print("Fetching Hacker News stories...")
    stories = fetch_hacker_news()
    
    if top_n:
        print(f"\nLimiting to top {top_n} stories")
        stories = stories[:top_n]
    
    print("\nProcessing stories...")
    all_summaries = []
    for i, story in enumerate(stories, 1):
        print(f"\nProcessing story {i}/{len(stories)}")
        story_content = get_hacker_news_story(story, max_tokens)
        story['content'] = story_content
        
        if story_content:
            print(f"Generating summary...:{story_content}")
            summary = summarize_content(story_content)
            story['summary'] = summary
            all_summaries.append(summary)  # Collecting all summaries
            print(f"Summary generated for {story['title']}")
            print(f"\nSummary:")
            print(summary)
    
    # Concatenate all summaries into one text
    concatenated_summary = "\n\n".join(all_summaries)
    print(f"\nCombined Summary:\n{concatenated_summary}")

    # Save the concatenated summary to a text file
    timestamp = time.strftime("%Y%m%d_%H%M")
    date = time.strftime("%Y%m%d")
    if not os.path.exists(date):
        os.makedirs(date)
    stories_filename = f'{date}/hacker_news_stories_{timestamp}.json'
    speech_filename = f'{date}/hacker_news_stories_{timestamp}.mp3'

    # Synthesize the concatenated summary into one audio file
    text_to_speech(speech_filename, concatenated_summary)

    # Save stories to a JSON file
    save_to_json(stories_filename, stories)

    print("\n=== Story Processing Results ===")
    for story in stories:
        print(f"\nTitle: {story['title']}")
        print(f"URL: {story['url']}")
        print(f"Hacker News URL: {story['hacker_news_url']}")
        print(f"Points: {story['points']}")
        print(f"Summary: {story.get('summary', 'No summary available')}")
        print("-" * 50)
    print("\n=== Job completed ===")


def schedule_job(interval, max_tokens, top_n):
    print(f"Scheduling job to run every {interval} minute(s)...")
    #schedule.every(interval).minutes.do(job, max_tokens=max_tokens, top_n=top_n)
    job(max_tokens, top_n)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Hacker News stories at a specified interval.")
    parser.add_argument('--interval', type=int, default=1, help='Interval in minutes to fetch stories')
    parser.add_argument('--max-tokens', type=int, default=1000, help='Maximum token length for story content')
    parser.add_argument('--top-n', type=int, default=10, help='Number of top stories to fetch (default: all)')
    args = parser.parse_args()
    
    schedule_job(args.interval, args.max_tokens, args.top_n)