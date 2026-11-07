import { evaluateDesign } from "./model.js";
import { runSearch } from "./search.js";

let stop = false;

self.onmessage = async (event) => {
  const message = event.data;
  if (message.type === "cancel") {
    stop = true;
    return;
  }
  if (message.type === "reeval") {
    try {
      const rows = message.designs.map((design) => evaluateDesign(design, message.task, null, true));
      self.postMessage({ type: "reeval", rows });
    } catch (error) {
      self.postMessage({ type: "error", message: error instanceof Error ? error.message : String(error) });
    }
    return;
  }
  if (message.type !== "start") return;
  stop = false;
  try {
    const result = await runSearch(message.spec, message.design, {
      yield: true,
      shouldStop: () => stop,
      onProgress: (record) => self.postMessage({ type: "progress", ...record, cancelled: stop }),
    });
    self.postMessage({ type: "done", result, cancelled: stop });
  } catch (error) {
    self.postMessage({ type: "error", message: error instanceof Error ? error.message : String(error) });
  }
};
