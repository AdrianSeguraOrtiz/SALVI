module.exports = {
  root: true,
  env: { browser: true, es2022: true },
  extends: [
    "eslint:recommended",
    "plugin:@typescript-eslint/recommended",
    "plugin:react-hooks/recommended"
  ],
  ignorePatterns: ["../src/salvi/web/static"],
  parser: "@typescript-eslint/parser",
  plugins: ["react-refresh"],
  parserOptions: {
    ecmaVersion: "latest",
    sourceType: "module"
  },
  rules: {
    "react-refresh/only-export-components": ["warn", { allowConstantExport: true }]
  }
};
