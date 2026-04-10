import esbuild from "esbuild";
import process from "process";
import { copyFileSync, mkdirSync, existsSync } from "fs";

const prod = process.argv[2] === "production";

if (!existsSync("dist")) mkdirSync("dist");
if (existsSync("src/styles.css")) copyFileSync("src/styles.css", "dist/styles.css");

const ctx = await esbuild.context({
  entryPoints: ["src/main.ts"],
  bundle: true,
  external: ["obsidian", "electron", "@codemirror/*", "@lezer/*"],
  format: "cjs",
  target: "es2018",
  logLevel: "info",
  sourcemap: prod ? false : "inline",
  treeShaking: true,
  outfile: "dist/main.js",
});

if (prod) {
  await ctx.rebuild();
  process.exit(0);
} else {
  await ctx.watch();
}
