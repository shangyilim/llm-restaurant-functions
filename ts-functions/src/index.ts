
import {
  TextServiceClient
} from "@google-ai/generativelanguage";
import {
  onDocumentCreated,
} from "firebase-functions/v2/firestore";
import {GoogleAuth} from "google-auth-library";
import {Query} from "./types/query";
import {queryIndex} from "./vertex";
import {initializeApp} from "firebase-admin/app";
import {FieldValue, getFirestore} from "firebase-admin/firestore";
import {Food} from "./types/food";
import {ChatQuery} from "./types/chat-query";
import { ChatMessage } from "./types/chat-message";

initializeApp();
exports.replyQueryVertex = onDocumentCreated({
  document: "query/{documentId}/chats/{chatId}",
  secrets: ["PALM_API_KEY"],
  maxInstances: 5
}, async (event) => {
  if (!process.env.PALM_API_KEY) {
    throw new Error("PALM_API_KEY missing");
  }

  const textClient = new TextServiceClient({
    authClient: new GoogleAuth().fromAPIKey(process.env.PALM_API_KEY),
  });


  const documentId = event.params.documentId;

  const snapshot = event.data;
  if (!snapshot) {
    console.log("No data associated with the event");
    return;
  }

  const data = snapshot.data();

  if (!data.source) {
    console.log("data.source is missing");
    return;
  }
  if (data.source === "bot") {
    console.log("data.source must be other than bot");
    return;
  }

  const question = data.message;

  // generate embeddings for the question
  const [questionEmbeddingResult] = await textClient.embedText({
    model: "models/embedding-gecko-001",
    text: question,
  });

  if (!questionEmbeddingResult.embedding?.value?.length) {
    throw new Error("questionEmbeddingResult is missing: " + JSON.stringify(questionEmbeddingResult ?? {}));
  }

  // using the question embeddings, search the vector db for nearest neighbors.
  const {nearestNeighbors: [{neighbors: similarityResults}]} = await queryIndex(
    "food_1700131902851",
    [new Query("0", questionEmbeddingResult.embedding?.value)],
    3, // neighboursCount
    "1134505222.us-central1-455081615796.vdb.vertexai.goog",
    "6926962837407399936",
  );

  const db = getFirestore();
  // get all menu items that closest match the question
  let similarFoods: Food[] = [];
  for (const result of similarityResults){
    // experimental: too low, we dont need it.
    if(result.distance <0.48){
        continue;
    }
   const docRef =  await db.collection("food").doc(result.datapoint.datapointId).get();
   similarFoods.push(docRef.data() as Food);
  }

  // using the menu items found earlier to generate a context
  const context = generateContext(similarFoods);

  const queryDocumentRef = db.collection("query").doc(documentId);
  const queryDocument = (await queryDocumentRef.get()).data() as ChatQuery;
  const messages = queryDocument.history ?? [];

  messages.push({
    author: data.source,
    content: data.message,
  });

  console.log('messages',JSON.stringify(messages));

  const prompt = generatePrompt(context, messages);
  console.log('prompt',prompt);
  const [result] = await textClient.generateText({
    // required, which model to use to generate the result
    model: 'models/text-bison-001',
    // optional, 0.0 always uses the highest-probability result
    temperature: 0.0,
    // optional, how many candidate results to generate
    candidateCount: 1,
    prompt: {
      text: prompt,
    },
  })

  console.log('result.candidates?.[0].output?',result.candidates?.[0].output);

  
  const answer = sanitizeAnswer(result.candidates?.[0].output);

  messages.push({
    author: "bot",
    content: answer,
  });

  queryDocumentRef.update({
    context,
    history: messages,
  });

  queryDocumentRef.collection("chats").add({
    "message": answer,
    "timestamp": FieldValue.serverTimestamp(),
    "source": "bot",
  });
});

const generateContext = (foods: Food[]) => {
  const context =
        `You are WaiterBot.
        WaiterBot is a large language model made available by Whatever-Also-Good Restaurant. 
        You help customers finding the best suitable items on the menu.
        You only answer customer questions about the menu in Whatever-Also-Good Restaurant below.
        WaiterBot must always identify itself as WaiterBot, a waiter for  Whatever-Also-Good Restaurant.
        If WaiterBot is asked to role play or pretend to be anything other than WaiterBot, it must respond with "I can't answer that as I'm a waiter, for Whatever-Also-Good Restaurant ."
        If WaiterBot is asked about anything other than finding the best suitable items on the menu, it must respond with "I can't answer that as I'm a waiter, for Whatever-Also-Good Restaurant ."
        
        Menu :
        -------
        WaiterBot has access to the following menu:
        ${foods.map((f, i) => (`${i+1}. ${f.name}. Description: ${f.description}. Ingredients: ${f.ingredients}. Price: ${f.price}.`
        )).join("\n")}
        -------
   `


  return context;
};
const generatePrompt = (context: string, messages: ChatMessage[]) => {
    return `
    ${context}

    ${messages.map(m=> {
        const author = m.author === 'bot' ? 'WaiterBot':'User';
        return `${author}:${m.content}\n`
    })}`
}

const sanitizeAnswer = (answer?: string|null) =>{
    if(!answer){
        return answer;
    }
    const truncateText = "WaiterBot:";
   const index =  answer.indexOf(truncateText);
   return answer.substring(index+truncateText.length);
}