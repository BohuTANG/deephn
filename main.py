from dataclasses import dataclass
from typing import List, Optional
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import os
import requests
from bs4 import BeautifulSoup
import json
import argparse
import openai
from urllib.parse import urljoin
import azure.cognitiveservices.speech as speechsdk
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Story:
    id: str
    title: str
    url: str
    hacker_news_url: str
    points: str
    comments: str
    content: Optional[str] = None
    summary: Optional[str] = None

    @property
    def formatted_content(self) -> str:
        parts = [f"<title>\n{self.title}\n</title>" if self.title else '',
                 f"<article>\n{self.content}\n</article>" if self.content else '',
                 f"<comments>\n{self.comments}\n</comments>" if self.comments else '']
        return '\n\n---\n\n'.join(filter(None, parts))

def check_env_var(env_var_name: str, error_message: str) -> str:
    value = os.getenv(env_var_name)
    if not value:
        raise ValueError(f"{error_message} is not set")
    return value

def summarize_content(content: str) -> str:
    PROMPT = """
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
            messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": content}],
            max_tokens=1000, temperature=0.7,
            extra_headers={"HTTP-Referer": "https://news.ycombinator.com", "X-Title": "HN Podcast Assistant"}
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error summarizing content: {str(e)}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_hacker_news_story(story: Story, max_tokens: int) -> str:
    logger.info(f"Fetching content for story: {story.title}")
    headers = {'X-Retain-Images': 'none'}
    story_content, comments_content = '', ''

    try:
        logger.info(f"Fetching article from: {story.url}")
        article_response = requests.get(f"https://r.jina.ai/{story.url}", headers=headers, timeout=30)
        article_response.raise_for_status()
        story_content = article_response.text[:max_tokens * 4]

        logger.info(f"Fetching comments from: {story.hacker_news_url}")
        comments_response = requests.get(
            f"https://r.jina.ai/{story.hacker_news_url}",
            headers={**headers, 'X-Remove-Selector': '.navs', 'X-Target-Selector': '#pagespace + tr'},
            timeout=30
        )
        comments_response.raise_for_status()
        comments_content = comments_response.text[:max_tokens * 4]
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching content: {str(e)}")
        raise

    story.content = story_content
    story.comments = comments_content
    return story.formatted_content

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_hacker_news() -> List[Story]:
    url = "https://news.ycombinator.com"
    logger.info(f"Fetching stories from {url}")
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        stories = []
        for item in soup.select('.athing'):
            try:
                title = item.select_one('.titleline > a').text
                story_url = item.select_one('.titleline > a')['href']
                story_id = item.get('id')
                hacker_news_url = f"https://news.ycombinator.com/item?id={story_id}"

                next_row = item.find_next_sibling('tr')
                points = next_row.select_one('.score')
                points = points.text if points else '0 points'
                comments = next_row.select_one('a[href^="item?id="]')
                comments = comments.text if comments else '0 comments'

                stories.append(Story(id=story_id, title=title, url=story_url, 
                                     hacker_news_url=hacker_news_url, points=points, comments=comments))
                logger.info(f"Found story: {title} ({points}, {comments})")
            except Exception as e:
                logger.error(f"Error parsing story: {str(e)}")
                continue
        return stories
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching stories: {str(e)}")
        raise

def save_to_json(file_name: str, stories: List[Story]):
    try:
        logger.info(f"Saving stories to {file_name}")
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump([vars(story) for story in stories], f, indent=4, ensure_ascii=False)
        logger.info("Stories saved successfully")
    except Exception as e:
        logger.error(f"Error saving stories: {str(e)}")
        raise

def text_to_speech(file_name: str, summary: str):
    try:
        speech_key = check_env_var('AZURE_SPEECH_KEY', 'Azure Speech key')
        service_region = check_env_var('AZURE_SPEECH_REGION', 'Azure Speech region')
        logger.info("Initializing speech synthesis...")

        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
        audio_config = speechsdk.audio.AudioOutputConfig(filename=file_name)
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

        logger.info(f"Starting speech synthesis for file {file_name}...")
        result = speech_synthesizer.speak_text_async(summary).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            logger.info(f"Speech synthesis completed successfully for {file_name}")
        else:
            logger.error(f"Speech synthesis failed. Reason: {result.reason}")
            raise Exception(f"Speech synthesis failed: {result.reason}")
    except Exception as e:
        logger.error(f"Error in text to speech conversion: {str(e)}")
        raise

def job(max_tokens: int, top_n: Optional[int] = None):
    logger.info("=== Starting Hacker News fetch job ===")
    try:
        stories = fetch_hacker_news()
        stories = stories[:top_n] if top_n else stories

        all_summaries = []
        for i, story in enumerate(stories, 1):
            try:
                logger.info(f"Processing story {i}/{len(stories)}")
                story_content = get_hacker_news_story(story, max_tokens)

                if story_content:
                    summary = summarize_content(story_content)
                    story.summary = summary
                    all_summaries.append(summary)
                    logger.info(f"Summary generated for {story.title}\nSummary:\n{summary}")
            except Exception as e:
                logger.error(f"Error processing story {story.title}: {str(e)}")
                continue

        output_dir = f"stories/{datetime.now().strftime('%Y-%m-%d')}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        save_to_json(f"{output_dir}/stories.json", stories)

        concatenated_summary = "\n\n".join(all_summaries)
        logger.info("Generating audio for combined summaries...")
        text_to_speech(f"{output_dir}/combined_summary.mp3", concatenated_summary)
    except Exception as e:
        logger.error(f"Error in job execution: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Hacker News stories at a specified interval.")
    parser.add_argument('--max-tokens', type=int, default=1000, help='Maximum token length for story content')
    parser.add_argument('--top-n', type=int, default=2, help='Number of top stories to fetch (default: all)')
    args = parser.parse_args()
    job(args.max_tokens, args.top_n)
