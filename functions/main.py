# Welcome to Cloud Functions for Firebase for Python!
# To get started, simply uncomment the below code or create your own.
# Deploy with `firebase deploy`

from firebase_functions import https_fn
from firebase_admin import (
   initialize_app,
   credentials,
   db,
   firestore,
)
import os

from firebase_functions.firestore_fn import (
  on_document_created,
  on_document_deleted,
  on_document_updated,
  on_document_written,
  Event,
  Change,
  DocumentSnapshot,
)

import google.generativeai as palm
import textwrap
import numpy as np
import pandas as pd
import chromadb
from chromadb.api.types import Documents, Embeddings

cred = credentials.Certificate("restaurant-llm-4-service-account.json")

initialize_app(cred, {
  "databaseURL": "https://restaurant-llm-4-default-rtdb.asia-southeast1.firebasedatabase.app"
})

chroma_client = chromadb.Client()

def get_model(modelType):
  models = [m for m in palm.list_models() if modelType in m.supported_generation_methods]
  return models[0]

def embed_function(texts: Documents) -> Embeddings:
  model = get_model("embedText")
  # Embed the documents using any supported method
  return  [palm.generate_embeddings(model=model, text=text)['embedding']
           for text in texts]

# # Get the embeddings of  text
def embed_fn(text):
  model = get_model("embedText")
  print("using:",model.name)
  embeddings =  palm.generate_embeddings(model=model, text=text)['embedding']
  print(embeddings)
  return embeddings

def storeEmbedding(id, text, embeddings):
   db.reference("embeddings/"+id).set({
      'text': text,
      'vectors': embeddings
   })



def make_prompt(context, query, relevant_passages):

  passage = ''
  for relevant_passage in relevant_passages:
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    passage = passage+ ". " + escaped
    
  prompt = ''
  if context == None:
    prompt = ("""You are a helpful waiter at a restaurant that answers questions from customers using the text from the reference menu included below. \
  Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
  Be sure to strike a friendly and conversational tone. \
  If the menu is irrelevant to the answer, you may ignore it.
  QUESTION: '{query}'
  MENU: '{passage}'
  """).format(query=query, passage=passage)
  else:
    prompt = ("""{context} \
              QUESTION: '{query}' \
              MENU: '{passage}'
              """).format(context=context, query=query, passage=passage)

  return prompt

def generate_prompt(chats, relevant_passages):
  passage = ''
  i = 1
  for relevant_passage in relevant_passages:
    passage = passage+ "\n" + str(i) +". "+relevant_passage
    i = i + 1
  
  chat_history = ''
  for chat in chats:
    source = "User" if chat['author'] == "user" else "WaiterBot"
    chat_history=("""{history}\n{source}:{message}""").format(source= source,message = chat['content'], history = chat_history)

  context = ("""You are WaiterBot.
        WaiterBot is a large language model made available by Whatever-Also-Good Restaurant. 
        You help customers finding the best suitable items on the menu.
        You only answer customer questions about the menu in Whatever-Also-Good Restaurant below.
        WaiterBot must always identify itself as WaiterBot, a waiter for  Whatever-Also-Good Restaurant.
        If WaiterBot is asked to role play or pretend to be anything other than WaiterBot, it must respond with "I can't answer that as I'm a waiter, for Whatever-Also-Good Restaurant ."
        If WaiterBot is asked about anything other than finding the best suitable items on the menu, it must respond with "I can't answer that as I'm a waiter, for Whatever-Also-Good Restaurant ."
        
        Menu :
        -------
        WaiterBot has access to the following menu:
        {passage}
        ------- 
        {history}
             """).format(passage=passage,history = chat_history)
  
  return context


@on_document_updated(
    document="food/{documentId}", 
    max_instances=1, 
    memory=256, secrets=["PALM_API_KEY"], 
    region= "asia-southeast1")
def generateEmbeddings(event: Event[DocumentSnapshot]) -> None:

  palm.configure(api_key=os.environ.get("PALM_API_KEY"))

  document_id = event.params['documentId']
  doc = event.data.after.to_dict()
  text = ("Food Name: %s . "
            "Description: %s . "
            "Ingredients: %s ."
            "Price: RM %s ."
            ) % (doc["name"], doc["description"], doc["ingredients"], doc["price"])
  embeddings = embed_fn(text)
  storeEmbedding(document_id, text, embeddings)

@on_document_created(
    document="query/{documentId}/chats/{chatId}", 
    max_instances=1,  
    memory=256,
    min_instances=1, 
    secrets=["PALM_API_KEY"], 
    region= "asia-southeast1")
def replyQuery(event: Event[DocumentSnapshot]) -> None:

  firestore_client = firestore.client()
  palm.configure(api_key=os.environ.get("PALM_API_KEY"))
  
  document_id = event.params['documentId']
  doc = event.data.to_dict()

  if doc.get('source') == None: 
    return
  
  if doc.get('source') == "bot":
    return


  question = doc['message']
  print(question)

  global chroma_client
  # start a in memory instance of chromadb
  if chroma_client == None:
    chroma_client = chromadb.Client()
  
  
  chroma_db = chroma_client.get_or_create_collection(name="foodmenu", embedding_function=embed_function)
  embedding_count = chroma_db.count()
  
  generated_embeddings = db.reference("embeddings").get()
  total_food = len(generated_embeddings)

  if embedding_count != total_food:
    # load all the stored embeddings

    # insert embeddings into the chroma database
    embeddings_arr = []
    documents_arr = []
    ids_arr=[] 
    for k in generated_embeddings:
        embeddings_arr.append(generated_embeddings[k]['vectors'])
        documents_arr.append(generated_embeddings[k]['text'])
        ids_arr.append(k)
    chroma_db.add(
      embeddings=embeddings_arr,
      documents=documents_arr,
      ids=ids_arr
    )

  # check context exist
  
  query_doc_ref = (firestore_client
               .collection("query")
               .document(document_id)
  )
  query = query_doc_ref.get().to_dict()

  
  prev_chats = query.get('history') or []
  prev_chats.append({
    "author": doc['source'],
    "content": doc['message']
  })

  print(prev_chats)
  # perform embedding search
  relevant_documents = chroma_db.query(query_texts=[question], n_results=5)['documents'][0]
  prompt = generate_prompt(prev_chats, relevant_documents)
  print(prompt)
  answer = palm.generate_text(
    model='models/text-bison-001', 
    prompt=prompt,
    candidate_count=1
  )

  answer_text = answer.result.replace("WaiterBot:","")

  query_doc_ref = (firestore_client
               .collection("query")
               .document(document_id)
  )
  
  chats_collection_ref = (firestore_client
                          .collection("query")
                          .document(document_id)
                          .collection("chats")
  )
  
  prev_chats.append({
    "author": 'bot',
    "content": answer_text
  })

  query_doc_ref.update({
    'context': prompt,
    'history': prev_chats,
  })

  chats_collection_ref.add({
    "message": answer_text,
    "timestamp": firestore.SERVER_TIMESTAMP,
    "source": "bot"
  })





    