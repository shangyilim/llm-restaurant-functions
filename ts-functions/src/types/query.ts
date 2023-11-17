/**
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

export class Query {
  id: string;
  featureVector: number[];

  constructor(id: string, featureVector: number[]) {
    this.id = id;
    this.featureVector = featureVector;
  }

  toVertexQuery() {
    return {
      datapoint: {
        datapoint_id: this.id,
        feature_vector: this.featureVector,
      },
    };
  }
}

export interface DataPoint {
    datapoint: {
        datapointId: string;
        crowdingTag: {
            crowdingAttribute: string
        }
    },
    distance: number,
}

export interface NearestNeighbors {
    nearestNeighbors: [{
        id: string,
        neighbors: DataPoint[],
    }]
}



