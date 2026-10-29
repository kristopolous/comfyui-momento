// Momento: https://9ol.es/momento-demo/ (any path containing "momento") opens the Momento page;
// add ?comfy to get the ComfyUI App Mode version of the same graph instead.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const here = location.pathname + location.search;
if (/momento/i.test(location.pathname) && !/[?&]comfy/.test(location.search)) {
  location.replace(location.pathname.replace(/\/?$/, "/") + "momento" + location.hash);
}

app.registerExtension({
  name: "Momento.App",
  async setup() {
    if (!/momento/i.test(here)) return;
    const res = await fetch(api.fileURL("/extensions/comfyui_momento/momento.json"));
    if (!res.ok) return;
    await app.loadGraphData(await res.json(), true, false, "Momento");
  },
});
