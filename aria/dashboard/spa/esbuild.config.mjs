import * as esbuild from 'esbuild';

const isWatch = process.argv.includes('--watch');

const config = {
  entryPoints: ['src/index.jsx'],
  bundle: true,
  outfile: 'dist/bundle.js',
  format: 'esm',
  jsx: 'transform',
  jsxFactory: 'h',
  jsxFragment: 'Fragment',
  inject: ['./src/preact-shim.js'],
  loader: { '.jsx': 'jsx' },
  minify: !isWatch,
  sourcemap: isWatch,
  logLevel: 'info',
};

if (isWatch) {
  const ctx = await esbuild.context(config);
  await ctx.watch();
  console.log('esbuild watching for changes...');
} else {
  await esbuild.build(config);
}
