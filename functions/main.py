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


initialize_app()


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

def generate_context(query, relevant_passages):
  passage = ''
  for relevant_passage in relevant_passages:
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    passage = passage+ ". " + escaped
    
  context = ("""You are a helpful waiter at a restaurant that answers questions from \
             customers using the text from the reference menu included below. Be sure to \
             respond in a complete sentence, being comprehensive, including all relevant \
             background information. Be sure to strike a friendly and conversational tone. \
             If the menu is irrelevant to the answer, you may ignore it. Only use information \
             provided below. Only talk abut the menu in the context. \
             Do not recommend food or drinks that is not part of the menu below. \
             ***
             MENU: 
             ***\
             '{passage}'""").format(query=query, passage=passage)
  
  return context


@on_document_written(document="food/{documentId}", max_instances=1, memory=512, secrets=["PALM_API_KEY"])
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

@on_document_created(document="query/{documentId}/chats/{chatId}", max_instances=1,  memory=512, secrets=["PALM_API_KEY"])
def replyQuery(event: Event[DocumentSnapshot]) -> None:

  palm.configure(api_key=os.environ.get("PALM_API_KEY"))
  
  document_id = event.params['documentId']
  doc = event.data.to_dict()

  if doc.get('source') == None: 
    return
  
  if doc.get('source') == "bot":
    return


  question = doc['message']
  print(question)

  # load all the stored embeddings
  all_embeddings = db.reference("embeddings").get()
  model_settings = db.reference("settings").get()

  # start a in memory instance of chromadb
  chroma_client = chromadb.Client()
  chroma_db = chroma_client.get_or_create_collection(name="foodmenu", embedding_function=embed_function)

  # insert embeddings into the chroma database
  embeddings_arr = []
  documents_arr = []
  ids_arr=[] 
  for k in all_embeddings:
      embeddings_arr.append(all_embeddings[k]['vectors'])
      documents_arr.append(all_embeddings[k]['text'])
      ids_arr.append(k)
  chroma_db.add(
    embeddings=embeddings_arr,
    documents=documents_arr,
    ids=ids_arr
  )

  # check context exist
  firestore_client = firestore.client()
  query_doc_ref = (firestore_client
               .collection("query")
               .document(document_id)
  )
  query = query_doc_ref.get().to_dict()

  
  prev_chats = query.get('history') or []
  # for chat_doc in chat_docs:
  #   prev_chat = chat_doc.to_dict()
  #   print(prev_chat)
  #   prev_chats.append({
  #     "author": prev_chat['source'],
  #     "content": prev_chat['message']})

  prev_chats.append({
    "author": doc['source'],
    "content": doc['message']
  })

  print(prev_chats)
  # perform embedding search
  relevant_documents = chroma_db.query(query_texts=[question], n_results=2)['documents'][0]
  context = generate_context(question, relevant_documents)
  print(context)
  answer = palm.chat(context=context, messages=prev_chats, candidate_count=1, temperature=model_settings['temperature'])


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
    "content": answer.last
  })

  query_doc_ref.update({
    'context': context,
    'history': prev_chats,
  })

  chats_collection_ref.add({
    "message": answer.last,
    "timestamp": firestore.SERVER_TIMESTAMP,
    "source": "bot"
  })





    

# def cosine_similarity(vector1, vector2):
#   cosine = np.dot(vector1,vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))
#   return cosine
  
# def find_similar_documents(query, type, count):
#   all_embeddings = db.reference("embeddings").get()
#   model = get_model()
#   query_embedding = palm.generate_embeddings(model=model, text=query)

#   top_similar_docs = []
#   vector_similarities = []
  
#   if( type == 'dotproduct'):
#     embeddings_arr = []
#     documents_dict = []
#     for k in all_embeddings:
#       embeddings_arr.append(all_embeddings[k]['vectors'])
#       documents_dict.append({
#         'key': k,
#         'text': all_embeddings[k]['text']
#       })
      
#     vector_similarities = np.dot(np.stack(embeddings_arr), query_embedding['embedding'])
#   else:
#     for k in all_embeddings:
#       vector_similarities.append(cosine_similarity(query_embedding, all_embeddings[k]['vectors']))
#       documents_dict.append({
#         'key': k,
#         'text': all_embeddings[k]['text']
#       })
  
#   for i in np.sort(vector_similarities)[::-1][:count]:
#     top_similar_docs.append(documents_dict[i])
  
#   return top_similar_docs
