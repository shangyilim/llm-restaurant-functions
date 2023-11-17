# Welcome to Cloud Functions for Firebase for Python!
# To get started, simply uncomment the below code or create your own.
# Deploy with `firebase deploy`

from firebase_functions import scheduler_fn
from firebase_admin import (
   initialize_app,
   firestore,
   storage,
)
import os
import json

from firebase_functions.firestore_fn import (
  Event,
)

import google.generativeai as palm

initialize_app()

@scheduler_fn.on_schedule(schedule="0 0 1 * *", max_instances=1, secrets=["PALM_API_KEY"])
def backfill(event: scheduler_fn.ScheduledEvent) -> None:
  palm.configure(api_key=os.environ.get("PALM_API_KEY"))

  firestore_client = firestore.client()
  docs = firestore_client.collection("food").stream()
  
  source_file_name = 'food_embeds.json'
  with open(source_file_name, 'a', encoding='utf-8') as f:
    for doc in docs:
      food = doc.to_dict()
      text = ("Food Name: %s . "
            "Description: %s . "
            "Ingredients: %s ."
            "Price: %s ."
            ) % (food["name"], food["description"], food["ingredients"], food["price"])
      embeddings = embed_fn(text)
      
      json_str = json.dumps({
        "id": doc.id,
        "embedding": embeddings
      })
      f.write(json_str+"\n")
  
  bucket = storage.bucket('restaurant-llm_embedding')
  blob = bucket.blob(source_file_name)
  blob.upload_from_filename(source_file_name)

