import js from "@eslint/js";
import globals from "globals";
import prettierConfig from "eslint-config-prettier";

/**
 * ESLint config for ARIA dashboard SPA (Preact + esbuild).
 * JSX parsed natively via ecmaFeatures.jsx = true.
 * @type {import('eslint').Linter.Config[]}
 */
export default [
  js.configs.recommended,
  prettierConfig,
  {
    files: ["src/**/*.{js,jsx}"],
    languageOptions: {
      globals: {
        ...globals.browser,
        h: "readonly", // Preact JSX factory (explicit imports + new transform)
      },
      parserOptions: {
        ecmaFeatures: { jsx: true },
      },
    },
    rules: {
      "no-unused-vars": ["warn", { argsIgnorePattern: "^_", varsIgnorePattern: "^_" }],
      "no-console": "off",
      "prefer-const": "error",
      "no-var": "error",
      // eqeqeq: warn only — existing codebase uses == extensively; enforce in new code
      eqeqeq: ["warn", "always"],
      "no-empty": "warn",
      // Declare as off so inline eslint-disable comments don't cause "rule not found" errors
      "react-hooks/exhaustive-deps": "off",
      // Downgrade to warn — pattern of let x = ''; if (...) x = ... is intentional
      "no-useless-assignment": "warn",
    },
  },
  {
    // scripts/ are Node.js utilities (screenshot-audit, etc.)
    files: ["scripts/**/*.js", "esbuild.config.mjs"],
    languageOptions: {
      globals: { ...globals.node, ...globals.browser },
    },
    rules: {
      "no-unused-vars": ["warn", { argsIgnorePattern: "^_" }],
      "prefer-const": "error",
    },
  },
  {
    ignores: ["node_modules/", "dist/"],
  },
];
