// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import type {ClientOptions} from 'google-gax';
import {PredictionServiceClient} from './v1/prediction_service_client';

/**
 * The result of a text embedding prediction, containing the embedding vector.
 */
export interface TextEmbedding {
  /** The embedding values as an array of numbers. */
  values: number[];
}

/**
 * Options for constructing a TextEmbeddingModel.
 */
export interface TextEmbeddingModelOptions {
  /** Google Cloud project ID. Defaults to the environment's project. */
  project?: string;
  /** Google Cloud location. Defaults to 'us-central1'. */
  location?: string;
  /** Additional options passed to the underlying PredictionServiceClient. */
  clientOptions?: ClientOptions;
}

/**
 * A high-level wrapper around Vertex AI text embedding models,
 * mirroring the Python `TextEmbeddingModel` interface.
 *
 * @example
 * const model = TextEmbeddingModel.fromPretrained('text-embedding-004');
 * const embeddings = await model.getEmbeddings(['What is life?']);
 * console.log(embeddings[0].values.slice(0, 5));
 * await model.close();
 */
export class TextEmbeddingModel {
  private readonly _modelName: string;
  private readonly _location: string;
  private readonly _project: string | undefined;
  private readonly _client: PredictionServiceClient;

  private constructor(
    modelName: string,
    options: TextEmbeddingModelOptions = {},
  ) {
    this._modelName = modelName;
    this._location = options.location ?? 'us-central1';
    this._project = options.project;
    this._client = new PredictionServiceClient({
      ...options.clientOptions,
      ...(options.project ? {projectId: options.project} : {}),
    });
  }

  /**
   * Creates a TextEmbeddingModel instance for the given pretrained model name.
   *
   * @param modelName - The name of the pretrained text embedding model,
   *   e.g. `'text-embedding-004'` or `'textembedding-gecko'`.
   * @param options - Optional configuration for project, location, and
   *   underlying client options.
   * @returns A new TextEmbeddingModel instance.
   */
  static fromPretrained(
    modelName: string,
    options?: TextEmbeddingModelOptions,
  ): TextEmbeddingModel {
    return new TextEmbeddingModel(modelName, options);
  }

  /**
   * Generates embeddings for each of the provided texts.
   *
   * @param texts - An array of strings to embed.
   * @returns An array of {@link TextEmbedding} objects, one per input text.
   */
  async getEmbeddings(texts: string[]): Promise<TextEmbedding[]> {
    const projectId = this._project ?? (await this._client.getProjectId());
    const endpoint = `projects/${projectId}/locations/${this._location}/publishers/google/models/${this._modelName}`;

    const instances = texts.map(text => ({
      structValue: {
        fields: {
          content: {stringValue: text},
        },
      },
    }));

    const [response] = await this._client.predict({endpoint, instances});

    const predictions = response.predictions ?? [];
    return predictions.map(prediction => {
      const fields =
        prediction?.structValue?.fields?.embeddings?.structValue?.fields?.values
          ?.listValue?.values ?? [];
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const values = fields.map((v: any) => (v.numberValue as number) ?? 0);
      return {values};
    });
  }

  /**
   * Releases resources held by the underlying gRPC client.
   * Call this when you are finished using the model.
   */
  async close(): Promise<void> {
    await this._client.close();
  }
}
