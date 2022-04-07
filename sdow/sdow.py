"""Solutions to six-degrees-of-wikipedia path finding."""

import plotext as plt
import click
import numpy as np
import random
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import cohere
import os

co = cohere.Client(os.environ['COHERE_API_TOKEN'])
pages_seen = set()


@click.command()
@click.argument('start', envvar='SDOW_START')
@click.argument('target', envvar='SDOW_TARGET')
@click.option('--classic', is_flag=True)
def sdow(start, target, classic):
    """Solutions to six-degrees-of-wikipedia path finding."""
    if classic:
        asyncio.run(classic_solution(start, target))
    else:
        asyncio.run(nlu_solution(start, target))


def nlu_solution(start, target):
    """NLU solution to six-degrees-of-wikipedia path finding."""
    return solve(start, target, nlu=True)


def classic_solution(start, target):
    """Classic solution to six-degrees-of-wikipedia path finding."""
    return solve(start, target, nlu=False)


async def solve(start, target, workers=1, nlu=False):
    """Task scheduling to solve six-degrees-of-wikipedia path finding."""
    fetch_queue = asyncio.Queue()
    await fetch_queue.put((0, start))
    async with aiohttp.ClientSession() as client:
        tasks = []
        for i in range(workers):
            task = asyncio.create_task(fetch_task(
                client, fetch_queue, target, nlu))
            tasks.append(task)
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in tasks:
            task.cancel()
        print(done)
        return done


async def fetch_task(client, q, target, nlu):
    """Fetch pages and enqueue discovered links until the target page is met."""
    while True:
        depth, page = await q.get()
        try:
            async with client.get(page) as response:
                print(f'Visiting: {page} Distance: {depth}')
                text = await response.read()
                if response.real_url in pages_seen:
                    continue

                page_links = find_page_links(text)
                if nlu:
                    page_links = nlu_sorted_page_links(page_links, target)
                else:
                    page_links = randomly_sorted_page_links(page_links)
                for page in page_links:
                    if page == target:
                        return depth + 1, page
                    if page not in pages_seen:
                        pages_seen.add(page)
                        q.put_nowait((depth + 1, page))
        except Exception as e:
            print(f'error with {page}: {e}')
            continue
        q.task_done()


def find_page_links(text):
    """Find all links within an HTML page."""
    soup = BeautifulSoup(text.decode('utf-8'), features="lxml")
    body = soup.find('body')
    page_links = set()
    for a in body.find_all('a', href=True):
        href = a['href']
        if href.startswith("/wiki/") and ':' not in href:
            clean_href = href.split("#")[0]
            discovered_url = "https://en.wikipedia.org" + clean_href
            page_links.add(discovered_url)
    return page_links


def randomly_sorted_page_links(page_links):
    """Sort a list of page links randomly."""
    return sorted(page_links, key=lambda v: (v, random.random()))


def nlu_sorted_page_links(page_links, target):
    """Sort a list of page links using natural language understanding."""
    texts = [target] + list(page_links)
    embeddings = co.embed(
        model='small', texts=texts).embeddings
    tgt = np.array(embeddings[0])
    candidates = np.array(embeddings[1:])
    similarities = zip(page_links, np.dot(tgt, np.transpose(candidates)))
    sorted_similarities = [(page, score) for page, score in sorted(
        similarities, key=lambda x: x[1])]
    visualize_nlu_sorting(embeddings, sorted_similarities)
    return [p for p, _ in sorted_similarities[::-1]]


def visualize_nlu_sorting(embeddings, similarities, limit=25):
    """Visualize embeddings and cosine similarity used for sorting."""
    target_embeddings = np.array(embeddings[0])
    abbreviated_similarities = [(p[30:], s) for p, s in similarities]
    pages = [p for p, s in abbreviated_similarities]
    scores = [s for p, s in abbreviated_similarities]
    top_scores = scores[::-1]
    top_pages = pages[::-1]
    plt.subplots(2, 3)

    # fixed chart view of top similarity scores
    plt.subplot(2, 1)
    plt.span(1, 1)
    plt.title(f"top {limit} similarity scores")
    plt.clc()
    plt.cld()
    plt.bar(top_pages, top_scores,
            orientation="vertical", width=0.001)

    for idx, _ in enumerate(embeddings):

        # streaming bar chart view of similarity scores
        plt.subplot(2, 2)
        plt.span(2, 1)
        plt.title("similarity scores for discovered page links")
        if limit+idx < len(pages):
            plt.clc()
            plt.cld()
            plt.cld()
            plt.bar(pages[idx:limit+idx], scores[idx:limit+idx],
                    orientation="horizontal", width=0.001)

        # streaming scatter plot view of target and candidate embedding vectors
        candidate_embeddings = np.array(embeddings[idx])
        plt.subplot(1, 1)
        plt.span(3, 1)
        plt.title("target text embedding (♥) vs. candidate text embeddings (•)")
        plt.clc()
        plt.clt()
        plt.cld()
        plt.scatter(target_embeddings, label="target", marker="heart")
        plt.scatter(candidate_embeddings,
                    label=f"candidate ({pages[idx-1]})", marker="dot", color="red")

        plt.show()
        plt.sleep(0.05)
