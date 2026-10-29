// Momento: opening ComfyUI at a path containing "momento" (e.g. https://9ol.es/momento-demo/) loads the Momento app.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
  name: "Momento.App",
  async setup() {
    if (!/momento/i.test(location.pathname + location.search)) return;
    const res = await fetch(api.fileURL("/extensions/comfyui_momento/momento.json"));
    if (!res.ok) return;
    await app.loadGraphData(await res.json(), true, true, "Momento");
  },
});
