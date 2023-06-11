#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import random

from xml.dom import minidom
import dicttoxml
from tqdm import tqdm
import arxiv

import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader

PROMPT = """以下の項目を出力せよ。翻訳が難しい技術用語については、日本語ではなく英単語を使用しても構わない。
```
キーワード:この論文のキーワードを最大5つ（この項目は英語で出力せよ）
要旨:論文の概要。どんなもの？課題は何？批判されている理論は何？なぜそれが問題なのか？どうそれを改善したのか？（日本語で200字程度）
入出力:入力と出力の例を示せ。入力はどんなもの？出力はどんなもの？画像であれば画像サイズやフレーム数など具体的に述べよ。（日本語で200字以内）
新規性・手法のキモ:この論文が提案する手法の核心となるアイデアと、具体的な説明（日本語で200字程度）
検証方法:どうやって提案アイディアが有効であることを示したのか？データセットや評価指標はなにか？具体的にどうやって評価したのか？結果はどうだったのか？（日本語で200字以内）
議論:検証結果から何が言えるか？逆に検証結果から言えないことは何か？今後の課題は何か？対象となるスコープにおいて網羅性と整合性はあるか？発展的な課題はあるか？（日本語で200字以内）
```"""

MODEL_NAME = 'gpt-4'
# MODEL_NAME = 'gpt-3.5-turbo'

# DOWNLOAD_PDF = False  # False if you have already downloaded the PDF
DOWNLOAD_PDF = True

TEMPERATURE = 0.0


def get_gpt_summary(paper_info):
    print('Summarizing paper: ', paper_info['title'])

    # Create vector store
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(paper_info['pages'], embedding=embeddings)
    # Create retriever
    retriever = db.as_retriever()

    # Create chat model
    llm = ChatOpenAI(temperature=TEMPERATURE, model_name=MODEL_NAME, openai_api_key=openai.api_key)
    crc = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, return_source_documents=True)
    query = PROMPT
    chat_history = []

    # Ask question
    print('Waiting for response from OpenAI server...')
    # ToDo: Sometimes, the response ended up with mid-sentence, so need to deal with it.
    response = crc({'question': query, 'chat_history': chat_history})

    print('Response received.')
    summary = response['answer']

    summary_dict = {}
    for b in summary.split('\n'):
        print('****', b)
        if b.startswith('キーワード'):
            summary_dict['keywords'] = b[6:].lstrip()
        elif b.startswith('要旨'):
            summary_dict['summary'] = b[3:].lstrip()
        elif b.startswith('入出力'):
            summary_dict['input_output'] = b[4:].lstrip()
        elif b.startswith('新規性・手法のキモ'):
            summary_dict['method'] = b[10:].lstrip()
        elif b.startswith('検証方法'):
            summary_dict['validation'] = b[5:].lstrip()
        elif b.startswith('議論'):
            summary_dict['discussion'] = b[3:].lstrip()
    print('Dict by GPT', summary_dict)
    return summary_dict


def get_paper_info(result, dirpath='./xmls', pdf_name='paper.pdf'):
    arxiv_info = {}
    arxiv_info['title'] = result.title
    arxiv_info['date'] = result.published.strftime('%Y-%m-%d')
    arxiv_info['time'] = result.published.strftime('%H:%M:%S')
    arxiv_info['authors'] = [x.name for x in result.authors]
    arxiv_info['year'] = str(result.published.year)
    arxiv_info['entry_id'] = str(result.entry_id)
    arxiv_info['primary_category'] = str(result.primary_category)
    arxiv_info['categories'] = result.categories
    arxiv_info['journal_ref'] = str(result.journal_ref)
    arxiv_info['pdf_url'] = str(result.pdf_url)
    arxiv_info['doi'] = str(result.doi)
    arxiv_info['abstract'] = str(result.summary)

    os.makedirs(dirpath, exist_ok=True)

    pdf_path = os.path.join(dirpath, pdf_name)
    print('Downloading: ', pdf_path)
    if DOWNLOAD_PDF:
        result.download_pdf(dirpath=dirpath, filename=pdf_name)
    else:
        print('Skip download.')
    arxiv_info['pdf'] = pdf_name

    loader = PyPDFLoader(os.path.join(dirpath, arxiv_info['pdf']))
    arxiv_info['pages'] = loader.load_and_split()

    gpt_info = get_gpt_summary(arxiv_info)

    paper_info = {'paper': {**arxiv_info, **gpt_info}}
    return paper_info


def main(query, output_dir='./xmls', num_papers=3, from_year=2017, max_results=100):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    result_list = []
    for arxiv_info in search.results():
        if arxiv_info.published.year >= from_year:
            result_list.append(arxiv_info)

    if len(result_list) <= 0:
        print('No result')
        sys.exit()

    # Sample papers if the sample number is less than searched papers
    results = random.sample(result_list, k=num_papers) if 0 < num_papers < len(
        result_list) else result_list

    os.makedirs(output_dir, exist_ok=True)
    for i, arxiv_info in tqdm(enumerate(results), total=len(results)):
        try:
            print(arxiv_info, arxiv_info.published.year, arxiv_info.title)
            arxiv_id = arxiv_info.entry_id.replace('http://', '').replace('/', '-')
            dir_path = os.path.join(output_dir, arxiv_id)

            # Get paper summary by GPT
            paper_summary = get_paper_info(arxiv_info, dirpath=dir_path)

            # Save paper info as xml
            paper_summary['paper']['query'] = query
            xml = dicttoxml.dicttoxml(paper_summary, attr_type=False, root=False).decode('utf-8')
            xml = repr(xml)[1:-1]  # Make xml raw string to ignore escape characters which cause parse error
            xml = minidom.parseString(xml).toprettyxml(indent='   ')
            xml = minidom.parseString(xml).toprettyxml(indent='   ')
            xml_path = os.path.join(dir_path, 'paper.xml')
            with open(xml_path, 'w') as f:
                f.write(xml)
                
        except Exception as e:
            print(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', '-y', type=int, help='from year', default=2017)
    parser.add_argument('--dir', '-d', type=str, help='destination', default='./xmls')
    parser.add_argument('--api_key', '-a', type=str, help='OpenAI API key')
    parser.add_argument('--max_n', '-m', type=int, help='max number of papers', default=100)
    parser.add_argument('--sample_n', '-n', type=int, help='sample number of papers', default=0)
    parser.add_argument('positional_args', nargs='+', help='query keywords')
    args = parser.parse_args()

    print(args)
    print(f'GPT model: {MODEL_NAME}')
    print(f'Model temperature: {TEMPERATURE}')
    print(f'Download PDF: {DOWNLOAD_PDF}')

    openai.api_key = args.api_key
    os.environ['OPENAI_API_KEY'] = openai.api_key

    main(query=f'all:%22 {" ".join(args.positional_args)} %22', num_papers=args.sample_n, from_year=args.year,
         output_dir=args.dir, max_results=args.max_n)
