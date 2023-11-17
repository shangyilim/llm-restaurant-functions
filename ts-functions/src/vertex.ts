

import axios from "axios";

import config from "./config";
import {NearestNeighbors, Query} from "./types/query";
import {getAccessToken} from "./utils";

export async function queryIndex(
  indexId: string,
  queries: Query[],
  searchResults: number,
  endpoint: string,
  indexEndpoint: string
): Promise<NearestNeighbors> {
  try {
    const accessToken = await getAccessToken();

    const response = await axios.post(
      `https://${endpoint}/v1beta1/projects/${config.vertexProjectId}/locations/${config.location}/indexEndpoints/${indexEndpoint}:findNeighbors`,
      {
        queries: queries.map((query) => query.toVertexQuery()),
        deployed_index_id: indexId,
        neighbor_count: searchResults,
      },
      {
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${accessToken}`,
        },
      }
    );

    console.log('response.data',JSON.stringify(response.data));
    return response.data;
  } catch (error) {
    console.error("Error calling the endpoint:", error);
    throw error;
  }
}
