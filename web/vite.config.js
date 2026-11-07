import { resolve } from "node:path";
import { defineConfig } from "vite";

// 项目 Pages 地址是 https://zhuyanhe1975-hit.github.io/zbot_mjlab_exolimb_grasp/
export default defineConfig({
  base: "/zbot_mjlab_exolimb_grasp/",
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
        design: resolve(__dirname, "design.html"),
      },
    },
  },
});
