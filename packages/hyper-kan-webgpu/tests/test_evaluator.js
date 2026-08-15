import assert from "node:assert";
import test from "node:test";
import { KanEvaluator, createDefaultSphereModel } from "../dist/index.js";

test("KanEvaluator 3D sphere evaluation and gradient", () => {
  const model = createDefaultSphereModel(0.5, 8, 4);
  const evaluator = new KanEvaluator(model);

  const valCenter = evaluator.evaluate3D(0, 0, 0);
  assert(typeof valCenter === "number");

  const grad = new Float32Array(3);
  evaluator.gradient3D(0.2, 0.3, -0.1, grad);
  assert.strictEqual(grad.length, 3);
  assert(!Number.isNaN(grad[0]));
  assert(!Number.isNaN(grad[1]));
  assert(!Number.isNaN(grad[2]));

  // Test online streaming update
  evaluator.updateOnlineStreaming(0.2, 0.3, -0.1, 1.0, 0.05);
  const valAfter = evaluator.evaluate3D(0.2, 0.3, -0.1);
  assert(!Number.isNaN(valAfter));

  // JSON export
  const exported = evaluator.exportJSON();
  assert.strictEqual(exported.rank, 8);
  assert.strictEqual(exported.spatial_dim, 3);
});
