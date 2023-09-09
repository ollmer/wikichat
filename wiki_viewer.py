import json
import re
import sys
import zipfile
from pathlib import Path
from urllib.parse import unquote

from flask import Flask, request
from markdown import markdown

root = Path("./wiki_bge_small_en_embeddings")
archive = zipfile.ZipFile(root / "data/en/paragraphs.zip", "r")
app = Flask(__name__)

def get_paragraphs_by_vec_idx(vec_idx):
    chunk_id = vec_idx // 100000
    line_id = vec_idx % 100000
    idmin = max(0, line_id - 100)
    idmax = line_id + 100
    id_set = set(range(idmin, idmax))
    paragraphs = []
    main_paragraph = None
    with archive.open(
        "enwiki_paragraphs_clean/enwiki_paragraphs_%03d.jsonl" % chunk_id
    ) as f:
        for i, l in enumerate(f):
            if i in id_set:
                paragraph = json.loads(l)
                paragraphs.append(paragraph)
            if i == line_id:
                main_paragraph = paragraph
            if i > idmax:
                break
    return paragraphs, main_paragraph


def render_page(paragraphs, main_paragraph):
    title = main_paragraph["page_title"]
    url = main_paragraph["page_url"]
    page_id = main_paragraph["page_id"]
    text = ""
    last_block_id = None
    last_sublock_id = None
    last_block_title = None
    for p in paragraphs:
        if p["page_id"] != page_id:
            continue
        titles, block_text = p["text"].split("\n", maxsplit=1)
        block_title = titles.split(". ", maxsplit=1)[1] if ". " in titles else ""
        if block_title == last_block_title:
            block_title = ""
        _, block_id, sublock_id = p["id"].split("_")
        if block_id != last_block_id or sublock_id != last_sublock_id:
            if len(block_title) > 0:
                text += f"\n\n## {block_title}\n\n"
                last_block_title = block_title
            else:
                text += "\n\n"
        text += block_text
        last_block_id = block_id
        last_sublock_id = sublock_id
    text = re.sub("\n+", "\n\n", text)
    return text, title, url


def get_page(vec_idx):
    paragraphs, main_paragraph = get_paragraphs_by_vec_idx(vec_idx)
    return render_page(paragraphs, main_paragraph)


@app.route("/")
def hello_world():
    vec_idx = request.args.get("vec_idx")
    try:
        vec_idx = int(vec_idx)
    except:
        vec_idx = None
    if vec_idx is None:
        markdown_page = "Page not found"
        title = "Page not found"
        url = ""
    else:
        markdown_page, title, url = get_page(vec_idx)
    online_link = (
        f'<div>Online page: <a href="{url}">{unquote(url)}</a></div>'
        if vec_idx is not None
        else ""
    )
    style = ".markdown-body { width: 50%}"
    prefix = f"""
        <!DOCTYPE html><html lang="en">
        <head>
            <meta charset="utf-8">
            <style type="text/css">{style}</style>
            <title>{title}</title>
        </head>
        <body><h1>{title}</h1>{online_link}<div class="markdown-body">"""
    postfix = f"</div></body></html>"
    return prefix + markdown("[TOC]\n\n" + markdown_page, extensions=["toc"]) + postfix


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(sys.argv[1]))
