from quart import Quart, render_template, request, Response
import os
from openai import OpenAI
from openai import AsyncOpenAI
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

app = Quart(__name__)

load_dotenv()
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
persist_directory = 'database/'
docsearch = Chroma(embedding_function=embeddings, persist_directory=persist_directory)

@app.route('/', methods=['GET', 'POST'])
async def index():
    return await render_template('index.html')

def gen_prompt(docs, query) -> str:
    return f"""
Question: {query}
Context: {[doc.page_content for doc in docs]}
Answer:
"""

async def prompt(query):
    docs = docsearch.similarity_search(query, k=4)
    prompt = gen_prompt(docs, query)
    print(prompt)
    return prompt

async def stream(input_text):
    client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    stream = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": await prompt(input_text)
            }],
        stream=True,
    )
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

@app.route('/completion', methods=['GET', 'POST'])
async def completion_api():
    if request.method == "POST":
        data = await request.form
        input_text = data['input_text']
        return Response(stream(input_text), mimetype='text/event-stream')
    else:
        return Response(None, mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)
