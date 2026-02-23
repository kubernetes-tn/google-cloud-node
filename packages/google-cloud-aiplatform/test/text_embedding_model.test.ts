/*!
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {describe, it, beforeEach, afterEach} from 'mocha';
import * as assert from 'assert';
import * as sinon from 'sinon';

const aiplatform = require('../src');
const {TextEmbeddingModel} = aiplatform;

describe('TextEmbeddingModel', () => {
  let sandbox: sinon.SinonSandbox;

  beforeEach(() => {
    sandbox = sinon.createSandbox();
  });

  afterEach(() => {
    sandbox.restore();
  });

  it('is exported from the package', () => {
    assert.ok(TextEmbeddingModel);
  });

  it('has a static fromPretrained factory method', () => {
    assert.strictEqual(typeof TextEmbeddingModel.fromPretrained, 'function');
  });

  it('fromPretrained returns a TextEmbeddingModel instance', () => {
    const model = TextEmbeddingModel.fromPretrained('text-embedding-004', {
      project: 'test-project',
    });
    assert.ok(model);
    assert.strictEqual(typeof model.getEmbeddings, 'function');
    assert.strictEqual(typeof model.close, 'function');
    // Clean up client stub
    sandbox.stub(model['_client'], 'close').resolves();
    model.close();
  });

  it('getEmbeddings returns TextEmbedding objects with values arrays', async () => {
    const model = TextEmbeddingModel.fromPretrained('text-embedding-004', {
      project: 'test-project',
    });

    // Stub the underlying predict call
    const fakeEmbeddingValues = [0.1, 0.2, 0.3, 0.4, 0.5];
    const fakePrediction = {
      structValue: {
        fields: {
          embeddings: {
            structValue: {
              fields: {
                values: {
                  listValue: {
                    values: fakeEmbeddingValues.map(v => ({numberValue: v})),
                  },
                },
              },
            },
          },
        },
      },
    };

    sandbox
      .stub(model['_client'], 'predict')
      .resolves([{predictions: [fakePrediction]}, undefined, undefined]);
    sandbox.stub(model['_client'], 'close').resolves();

    const embeddings = await model.getEmbeddings(['What is life?']);

    assert.strictEqual(embeddings.length, 1);
    assert.ok(Array.isArray(embeddings[0].values));
    assert.deepStrictEqual(embeddings[0].values, fakeEmbeddingValues);

    await model.close();
  });

  it('getEmbeddings handles multiple texts', async () => {
    const model = TextEmbeddingModel.fromPretrained('text-embedding-004', {
      project: 'test-project',
    });

    const makePrediction = (vals: number[]) => ({
      structValue: {
        fields: {
          embeddings: {
            structValue: {
              fields: {
                values: {
                  listValue: {
                    values: vals.map(v => ({numberValue: v})),
                  },
                },
              },
            },
          },
        },
      },
    });

    sandbox.stub(model['_client'], 'predict').resolves([
      {
        predictions: [makePrediction([0.1, 0.2]), makePrediction([0.3, 0.4])],
      },
      undefined,
      undefined,
    ]);
    sandbox.stub(model['_client'], 'close').resolves();

    const embeddings = await model.getEmbeddings(['text 1', 'text 2']);

    assert.strictEqual(embeddings.length, 2);
    assert.deepStrictEqual(embeddings[0].values, [0.1, 0.2]);
    assert.deepStrictEqual(embeddings[1].values, [0.3, 0.4]);

    await model.close();
  });

  it('getEmbeddings uses correct endpoint format with project and location', async () => {
    const model = TextEmbeddingModel.fromPretrained('text-embedding-004', {
      project: 'my-project',
      location: 'europe-west4',
    });

    let capturedRequest: {endpoint?: string} = {};
    sandbox
      .stub(model['_client'], 'predict')
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      .callsFake((req: any) => {
        capturedRequest = req;
        return Promise.resolve([{predictions: []}, undefined, undefined]);
      });
    sandbox.stub(model['_client'], 'close').resolves();

    await model.getEmbeddings(['hello']);

    assert.strictEqual(
      capturedRequest.endpoint,
      'projects/my-project/locations/europe-west4/publishers/google/models/text-embedding-004',
    );

    await model.close();
  });

  it('getEmbeddings defaults to us-central1 when no location specified', async () => {
    const model = TextEmbeddingModel.fromPretrained('text-embedding-004', {
      project: 'my-project',
    });

    let capturedRequest: {endpoint?: string} = {};
    sandbox
      .stub(model['_client'], 'predict')
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      .callsFake((req: any) => {
        capturedRequest = req;
        return Promise.resolve([{predictions: []}, undefined, undefined]);
      });
    sandbox.stub(model['_client'], 'close').resolves();

    await model.getEmbeddings(['hello']);

    assert.ok(capturedRequest.endpoint?.includes('/locations/us-central1/'));

    await model.close();
  });
});
